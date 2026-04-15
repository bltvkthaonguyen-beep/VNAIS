"""
Gene2Care-AI — Clinical Decision Support Demo  (v3)
====================================================
Run:  streamlit run main.py
Requires: streamlit, pandas, numpy, joblib, scikit-learn, matplotlib
Optional: shap  (pip install shap)

Required file (one of):
  gene2care_combined.joblib  -- all models + extras in one file  (preferred)
  OR both: gene2care_models.joblib + gene2care_extras.joblib

New in v3
----------
1. SHAP Explainability  — global bar chart + local force/waterfall per patient
   Falls back to permutation importance if SHAP is not installed.
2. Probability Calibration — reliability diagram (calibration curve)
   for all 3 risk classes, Brier score comparison uncal vs calibrated.
3. Longitudinal / Time-series modeling — simulate treatment response
   trajectory across 4 assessment timepoints; animated trend chart.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64
import streamlit as st

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
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.top-banner{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);border-radius:14px;padding:22px 30px;margin-bottom:22px;display:flex;align-items:center;gap:16px;}
.top-banner h1{color:#e8f4f8;font-size:23px;font-weight:600;margin:0;letter-spacing:-.3px;}
.top-banner p{color:#94b8c4;font-size:12px;margin:3px 0 0;}
.badge{background:rgba(255,255,255,.08);color:#7ecef5;font-size:10px;font-family:'DM Mono',monospace;padding:2px 8px;border-radius:20px;border:1px solid rgba(126,206,245,.25);display:inline-block;margin-top:5px;}
.sec{font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#94a3b8;margin:20px 0 8px;padding-bottom:5px;border-bottom:1px solid #e8edf2;}
.score-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:13px 15px;margin-bottom:8px;}
.score-card .lbl{font-size:10px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;}
.score-card .val{font-size:25px;font-weight:600;color:#1e293b;font-family:'DM Mono',monospace;line-height:1;}
.score-card .bo{background:#e2e8f0;border-radius:4px;height:5px;margin-top:6px;overflow:hidden;}
.score-card .bi{height:5px;border-radius:4px;}
.risk-medium{background:#fffbeb;border:1.5px solid #fcd34d;border-radius:10px;padding:13px 17px;}
.risk-low{background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;padding:13px 17px;}
.risk-label{font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;opacity:.6;}
.risk-medium .risk-value{font-size:19px;font-weight:600;color:#d97706;margin-top:3px;}
.risk-low .risk-value{font-size:19px;font-weight:600;color:#16a34a;margin-top:3px;}
@keyframes pulse-border{0%{border-color:#fca5a5;box-shadow:0 0 0 0 rgba(239,68,68,.25);}50%{border-color:#ef4444;box-shadow:0 0 0 6px rgba(239,68,68,0);}100%{border-color:#fca5a5;box-shadow:0 0 0 0 rgba(239,68,68,.25);}}
.risk-high-banner{background:#fef2f2;border:2px solid #fca5a5;border-radius:12px;padding:17px 20px;animation:pulse-border 2s ease-in-out infinite;}
.risk-high-banner .rh-label{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#b91c1c;opacity:.8;}
.risk-high-banner .rh-title{font-size:21px;font-weight:700;color:#dc2626;margin:4px 0;}
.risk-high-banner .rh-sub{font-size:12px;color:#7f1d1d;line-height:1.5;margin-top:5px;}
.risk-high-banner .rh-cl{margin-top:11px;border-top:1px solid #fecaca;padding-top:9px;}
.risk-high-banner .rh-check{font-size:12px;color:#991b1b;margin:3px 0;display:flex;gap:8px;align-items:flex-start;}
.conf-wrap{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:13px 15px;}
.conf-title{font-size:10px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;}
.conf-row{display:flex;align-items:center;gap:9px;margin-bottom:7px;}
.conf-lbl{font-size:12px;font-weight:500;width:60px;}
.conf-bo{flex:1;background:#e2e8f0;border-radius:3px;height:7px;overflow:hidden;}
.conf-bar-high{height:7px;border-radius:3px;background:#ef4444;}
.conf-bar-medium{height:7px;border-radius:3px;background:#f59e0b;}
.conf-bar-low{height:7px;border-radius:3px;background:#22c55e;}
.conf-pct{font-size:12px;font-family:'DM Mono',monospace;color:#64748b;width:36px;text-align:right;}
.conf-foot{font-size:11px;color:#64748b;margin-top:7px;padding-top:7px;border-top:1px solid #e2e8f0;}
.explain-item{display:flex;align-items:flex-start;gap:10px;padding:9px 13px;margin-bottom:5px;background:#f8fafc;border-radius:8px;border-left:3px solid #3b82f6;font-size:12px;color:#334155;line-height:1.55;}
.explain-item.warn{border-left-color:#f59e0b;background:#fffdf5;}
.explain-item.good{border-left-color:#22c55e;background:#f9fef9;}
.explain-item.alert{border-left-color:#ef4444;background:#fff8f8;}
.explain-item.model{border-left-color:#8b5cf6;background:#faf8ff;}
.explain-item.info{border-left-color:#3b82f6;}
.cmp-row{display:flex;align-items:center;justify-content:space-between;padding:9px 13px;border-radius:8px;margin-bottom:5px;font-size:12px;}
.cmp-match{background:#f0fdf4;border:1px solid #bbf7d0;}
.cmp-differ{background:#fff7ed;border:1px solid #fed7aa;}
.cmp-label{color:#64748b;font-size:10px;font-weight:600;text-transform:uppercase;}
.cmp-val-ml{font-weight:600;color:#1e293b;}
.cmp-val-rb{color:#64748b;font-family:'DM Mono',monospace;font-size:11px;}
.model-differs{background:#fff7ed;border:1.5px solid #fdba74;border-radius:10px;padding:11px 15px;margin:12px 0;font-size:12px;color:#7c2d12;}
.model-agrees{background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;padding:11px 15px;margin:12px 0;font-size:12px;color:#14532d;}
.scenario-label{font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.07em;margin:12px 0 5px;}
.disclaimer{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:9px 13px;font-size:10px;color:#94a3b8;line-height:1.6;margin-top:18px;}
.metric-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 15px;text-align:center;}
.metric-card .mlbl{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:4px;}
.metric-card .mval{font-size:22px;font-weight:600;color:#1e293b;font-family:'DM Mono',monospace;}
.metric-card .msub{font-size:11px;color:#64748b;margin-top:2px;}
.traj-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:4px;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all():
    """
    Load strategy (tries in order):
    1. gene2care_combined.joblib  -- single merged file (preferred for deployment)
    2. gene2care_models.joblib + gene2care_extras.joblib  -- separate files (legacy)
    All keys from both files are merged into a single dict returned as (data, data).
    """
    # Try combined file first
    try:
        data = joblib.load("gene2care_combined.joblib")
        return data, data
    except FileNotFoundError:
        pass
    # Fallback: load separate files
    try:
        models = joblib.load("gene2care_models.joblib")
        extras = joblib.load("gene2care_extras.joblib")
        combined = {**models, **extras}
        return combined, combined
    except FileNotFoundError:
        return None, None

models, extras = load_all()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
NUMERIC      = ["age","phq9","gad7","sleep_quality","symptom_duration_months","recurrence_count"]
CATEGORICAL  = ["gender","prior_treatment_status","functional_impairment","therapy_access"]
FEATURE_NAMES = NUMERIC + CATEGORICAL
TARGET_REG   = ["medication_first_suitability","psychotherapy_first_suitability","combined_care_suitability"]
CLASSES      = ["high","low","medium"]

SCORE_LABELS = {"medication_first_suitability":"Medication First",
                "psychotherapy_first_suitability":"Psychotherapy First",
                "combined_care_suitability":"Combined Care"}
SCORE_COLORS = {"medication_first_suitability":"#3b82f6",
                "psychotherapy_first_suitability":"#8b5cf6",
                "combined_care_suitability":"#06b6d4"}
RISK_COLORS  = {"high":"#ef4444","medium":"#f59e0b","low":"#22c55e"}

TIMEPOINT_LABELS = ["T0 — Baseline","T1 — Week 2","T2 — Week 4","T3 — Week 8"]

# ──────────────────────────────────────────────────────────────────────────────
# DEMO SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────
DEMO_SCENARIOS = {
    "🌱 First Episode": {
        "desc":"Young adult, first episode, moderate depression, high therapy access",
        "age":20,"gender":"Female","phq9":14,"gad7":8,"sleep_quality":2,
        "symptom_duration_months":4,"recurrence_count":0,
        "prior_treatment_status":"none","functional_impairment":"mild","therapy_access":"high"},
    "😰 Severe + Anxious": {
        "desc":"Severe depression, high anxiety, poor sleep, partial prior treatment",
        "age":27,"gender":"Female","phq9":23,"gad7":18,"sleep_quality":5,
        "symptom_duration_months":10,"recurrence_count":1,
        "prior_treatment_status":"medication_only","functional_impairment":"severe","therapy_access":"medium"},
    "🔁 Recurrent Chronic": {
        "desc":"Chronic, 3 relapses, failed prior treatment — HIGH non-response risk",
        "age":36,"gender":"Male","phq9":19,"gad7":14,"sleep_quality":4,
        "symptom_duration_months":36,"recurrence_count":3,
        "prior_treatment_status":"failed_or_discontinued","functional_impairment":"moderate","therapy_access":"low"},
    "🚧 Low Access": {
        "desc":"Moderate depression, no therapy access — access-driven medication uplift",
        "age":29,"gender":"Male","phq9":17,"gad7":11,"sleep_quality":4,
        "symptom_duration_months":9,"recurrence_count":1,
        "prior_treatment_status":"none","functional_impairment":"moderate","therapy_access":"low"},
}

# ──────────────────────────────────────────────────────────────────────────────
# RULE-BASED BASELINE
# ──────────────────────────────────────────────────────────────────────────────
def get_phq9_band(p):
    if p<=9: return "mild"
    if p<=14: return "moderate"
    if p<=19: return "moderately_severe"
    return "severe"

def get_gad7_band(g):
    if g<=4: return "minimal"
    if g<=9: return "mild"
    if g<=14: return "moderate"
    return "severe"

def rule_based_predict(inp):
    phq9=inp["phq9"]; gad7=inp["gad7"]; sleep_q=inp["sleep_quality"]
    dur=inp["symptom_duration_months"]; rel=inp["recurrence_count"]
    fi=inp["functional_impairment"]; ta=inp["therapy_access"]; ptx=inp["prior_treatment_status"]
    pb=get_phq9_band(phq9); gb=get_gad7_band(gad7)
    fis=(fi=="severe"); failed=(ptx=="failed_or_discontinued"); chronic=(dur>12)
    if failed:                           med="🔄 Switch / Augment"
    elif pb in("moderately_severe","severe"): med="💊 SSRI First-Line"
    elif rel>=1 and phq9>14:             med="💊 SSRI or SNRI"
    else:                                med="⏳ Not Primary"
    if ta=="low":                        ther="📱 Digital / Self-Guided CBT"
    elif pb=="mild" and gb in("minimal","mild"): ther="🌱 Low-Intensity CBT"
    elif gb in("moderate","severe"):     ther="🎯 CBT with Anxiety Focus"
    elif chronic:                        ther="🔁 CBT + Behavioural Activation"
    else:                                ther="🗣️ Standard CBT"
    sev=(phq9/27)*4+(gad7/21)*2+(sleep_q-1)/4
    combined="✅ Combined Care Indicated" if (sev>5 or rel>=2 or fis) else "➖ Single Modality"
    rp=0
    if chronic: rp+=1
    if rel>=1:  rp+=1
    if failed:  rp+=2
    if sleep_q>=4: rp+=1
    if fis:     rp+=1
    if phq9>=20: rp+=1
    risk="high" if rp>=4 else ("medium" if rp>=2 else "low")
    return {"medication":med,"therapy":ther,"combined_care":combined,"risk":risk,"risk_pts":rp}

# ──────────────────────────────────────────────────────────────────────────────
# ML PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def ml_predict(inp, models, extras):
    if models is None: return None
    row    = pd.DataFrame([{k:inp[k] for k in FEATURE_NAMES}])
    rf_clf = models["rf_clf"]; rf_reg = models["rf_reg"]
    risk_label = rf_clf.predict(row)[0]
    risk_proba = dict(zip(rf_clf.classes_, rf_clf.predict_proba(row)[0].round(3)))
    scores = {t:round(float(s),1) for t,s in zip(TARGET_REG, rf_reg.predict(row)[0])}

    # Calibrated probability
    cal_proba = None
    if extras and "cal_pipeline" in extras:
        cal_proba = dict(zip(extras["cal_pipeline"].classes_,
                             extras["cal_pipeline"].predict_proba(row)[0].round(3)))

    ms=scores["medication_first_suitability"]
    ps=scores["psychotherapy_first_suitability"]
    cs=scores["combined_care_suitability"]

    if inp["prior_treatment_status"]=="failed_or_discontinued": ml="🔄 Switch / Augment"
    elif ms>=65: ml="💊 SSRI First-Line"
    elif ms>=45: ml="💊 SSRI or SNRI"
    else:        ml="⏳ Not Primary"

    if ps>=65:
        tl=("📱 Digital / Self-Guided CBT" if inp["therapy_access"]=="low"
            else ("🎯 CBT with Anxiety Focus" if inp["gad7"]>=15 else "🗣️ Standard CBT"))
    elif ps>=45: tl="🔁 CBT + Behavioural Activation"
    else:        tl="🌱 Low-Intensity CBT"

    cl="✅ Combined Care Indicated" if cs>=65 else "➖ Single Modality"

    return {"medication":ml,"therapy":tl,"combined_care":cl,
            "risk":risk_label,"risk_proba":risk_proba,
            "cal_proba":cal_proba,"scores":scores}

# ──────────────────────────────────────────────────────────────────────────────
# EXPLANATION
# ──────────────────────────────────────────────────────────────────────────────
def build_explanation(inp, ml_result):
    items = []
    phq9=inp["phq9"]; gad7=inp["gad7"]; sq=inp["sleep_quality"]
    dur=inp["symptom_duration_months"]; rel=inp["recurrence_count"]
    fi=inp["functional_impairment"]; ptx=inp["prior_treatment_status"]; ta=inp["therapy_access"]
    hs=ml_result is not None and "scores" in ml_result
    ms=ml_result["scores"]["medication_first_suitability"] if hs else None
    ps=ml_result["scores"]["psychotherapy_first_suitability"] if hs else None
    cs=ml_result["scores"]["combined_care_suitability"] if hs else None
    risk=ml_result.get("risk") if ml_result else None
    proba=ml_result.get("cal_proba") or ml_result.get("risk_proba",{}) if ml_result else {}

    # Clinical
    if phq9>=20:
        items.append({"source":"clinical","tone":"alert","text":
            f"PHQ-9 = {phq9} (severe) — DSM criteria indicate pharmacotherapy and close monitoring."})
    elif phq9>=15:
        items.append({"source":"clinical","tone":"warn","text":
            f"PHQ-9 = {phq9} (moderately severe) — structured treatment strongly recommended."})
    else:
        items.append({"source":"clinical","tone":"good","text":
            f"PHQ-9 = {phq9} (moderate) — psychotherapy-first is clinically viable."})
    if gad7>=15:
        items.append({"source":"clinical","tone":"alert","text":
            f"GAD-7 = {gad7} (severe anxiety) — independently elevates non-response risk; anxiety-focused CBT or combined care indicated."})
    elif gad7>=10:
        items.append({"source":"clinical","tone":"warn","text":
            f"GAD-7 = {gad7} (moderate comorbid anxiety) — anxious depression responds better to combined treatment."})
    if sq>=4:
        items.append({"source":"clinical","tone":"alert","text":
            f"Sleep quality = {sq}/5 (very poor) — validated predictor of non-response; CBT-I alongside main treatment may improve outcomes."})
    elif sq<=2:
        items.append({"source":"clinical","tone":"good","text":
            f"Sleep quality = {sq}/5 (good) — protective factor for treatment response."})
    if dur>24:
        items.append({"source":"clinical","tone":"alert","text":
            f"Duration = {dur} months (chronic >2 yr) — reduced single-modality response; combined + maintenance indicated."})
    elif dur>12:
        items.append({"source":"clinical","tone":"warn","text":
            f"Duration = {dur} months — sub-chronic; treatment history relevant to sequencing."})
    if rel>=3:
        items.append({"source":"clinical","tone":"alert","text":
            f"{rel} prior episodes — high recurrence predicts future episodes; maintenance + relapse-prevention CBT warranted."})
    elif rel>=1:
        items.append({"source":"clinical","tone":"warn","text":
            f"{rel} prior episode(s) — increases risk; supports combined or stepped-care."})
    if ptx=="failed_or_discontinued":
        items.append({"source":"clinical","tone":"alert","text":
            "Prior treatment failed/discontinued — switch class (SNRI, atypical) or augmentation strategy rather than first-line SSRI."})
    elif ptx=="none":
        items.append({"source":"clinical","tone":"good","text":"No prior treatment — first-line approach appropriate."})
    if fi=="severe":
        items.append({"source":"clinical","tone":"alert","text":
            "Severe functional impairment — combined care and intensive monitoring strongly indicated."})
    elif fi=="mild":
        items.append({"source":"clinical","tone":"good","text":"Mild impairment — psychotherapy-first is viable."})
    if ta=="low":
        items.append({"source":"clinical","tone":"warn","text":
            "Low therapy access — structural barrier; medication or digital CBT may be most practical."})
    elif ta=="high":
        items.append({"source":"clinical","tone":"good","text":"High therapy access — structured CBT structurally feasible."})

    # Model score patterns
    if hs:
        top_t=max(ml_result["scores"],key=ml_result["scores"].get)
        items.append({"source":"model","tone":"info","text":
            f"Highest suitability: {SCORE_LABELS[top_t]} = {ml_result['scores'][top_t]:.0f}/100 — "
            f"driven by PHQ-9, GAD-7, sleep, chronicity, and access patterns in the training data."})
        if cs>=65:
            drivers=[f"PHQ-9={phq9}" if phq9>=18 else None,
                     f"GAD-7={gad7}" if gad7>=12 else None,
                     f"sleep={sq}/5" if sq>=4 else None,
                     f"{dur}mo duration" if dur>12 else None,
                     f"{rel} relapses" if rel>=2 else None]
            drivers=[d for d in drivers if d]
            if drivers:
                items.append({"source":"model","tone":"warn","text":
                    f"Combined care score = {cs:.0f}/100, elevated by: {', '.join(drivers)}. "
                    f"The model learned this combination predicts higher combined-treatment benefit."})
        if ps<45 and ta=="high":
            items.append({"source":"model","tone":"info","text":
                f"Psychotherapy score = {ps:.0f}/100 despite high access — severity/chronicity "
                f"suppresses the psychotherapy signal even when access is not a barrier."})
        if ms>=55 and ta=="low":
            items.append({"source":"model","tone":"warn","text":
                f"Medication score = {ms:.0f}/100 partly elevated by low therapy access — "
                f"structural artifact, not pure clinical optimality."})
        high_conf=proba.get("high",0)
        if risk=="high" and high_conf>=0.6:
            items.append({"source":"model","tone":"alert","text":
                f"High-risk confidence = {high_conf:.0%} — strong co-occurrence of "
                f"{'chronic duration, ' if dur>12 else ''}"
                f"{'multiple relapses, ' if rel>=2 else ''}"
                f"{'failed treatment, ' if ptx=='failed_or_discontinued' else ''}"
                f"poor sleep and severity in training data."})
        elif risk=="low" and proba.get("low",0)>=0.7:
            items.append({"source":"model","tone":"good","text":
                f"Low-risk confidence = {proba['low']:.0%} — absence of chronicity, recurrence, "
                f"and treatment failure creates a clear low-risk signal."})
        elif ml_result and proba.get(risk,0)<0.50:
            items.append({"source":"model","tone":"warn","text":
                f"Risk = {(risk or '').upper()} but confidence only {proba.get(risk,0):.0%} — "
                f"profile near a decision boundary; clinical judgement carries extra weight."})
    return items

# ──────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB → BASE64 helper
# ──────────────────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def show_fig(fig, caption=""):
    b64 = fig_to_b64(fig)
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;" />',
        unsafe_allow_html=True)
    if caption:
        st.caption(caption)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL SHAP / PERMUTATION IMPORTANCE CHART
# ──────────────────────────────────────────────────────────────────────────────
def plot_global_importance(extras, target_class="high", regression_target=None):
    """
    If SHAP is available (precomputed in extras), render beeswarm-style bar.
    Otherwise render permutation importance.
    regression_target: if set, show regression SHAP for that target.
    """
    shap_avail = extras.get("shap_available", False) if extras else False

    if shap_avail and extras:
        if regression_target:
            key = regression_target
            data = extras.get("shap_reg", {}).get(key, {})
            if not data: return None
            mean_abs = np.array(data["mean_abs"])
            title = f"Global SHAP — {SCORE_LABELS.get(key, key)}"
            color = SCORE_COLORS.get(key, "#3b82f6")
        else:
            data = extras.get("shap_global", {}).get(target_class, {})
            if not data: return None
            mean_abs = np.array(data["mean_abs"])
            title = f"Global SHAP — risk = {target_class}"
            color = RISK_COLORS.get(target_class, "#64748b")

        idx = np.argsort(mean_abs)
        feat_names = extras.get("FEATURE_NAMES", FEATURE_NAMES)
        labels = [feat_names[i] for i in idx]
        vals   = mean_abs[idx]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(labels, vals, color=color, alpha=0.82, height=0.6)
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="500")
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(val+0.0005, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=7.5, color="#334155")
        fig.tight_layout()
        return fig

    # Fallback: use pre-fitted RF feature_importances_
    if extras and "rf_raw" in extras:
        rf = extras["rf_raw"]
        imp = rf.feature_importances_
        idx = np.argsort(imp)
        labels = [FEATURE_NAMES[i] for i in idx]
        vals   = imp[idx]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(labels, vals, color="#94a3b8", alpha=0.85, height=0.6)
        ax.set_xlabel("RF Feature Importance (Gini)", fontsize=9)
        ax.set_title("Global Feature Importance (Gini impurity — SHAP not installed)",
                     fontsize=9, fontweight="500")
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        return fig
    return None


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL SHAP — waterfall for one patient
# ──────────────────────────────────────────────────────────────────────────────
def plot_local_shap(inp, extras, target_class="high"):
    if not extras or not extras.get("shap_available", False):
        return None
    shap_global = extras.get("shap_global", {})
    if target_class not in shap_global:
        return None

    preprocessor = extras["preprocessor"]
    row  = pd.DataFrame([{k:inp[k] for k in FEATURE_NAMES}])
    row_t = pd.DataFrame(preprocessor.transform(row), columns=FEATURE_NAMES)

    try:
        import shap
        rf_raw = extras["rf_raw"]
        expl   = shap.TreeExplainer(rf_raw)
        sv     = expl.shap_values(row_t)
        cidx   = list(rf_raw.classes_).index(target_class)
        shap_vals = sv[cidx][0]          # 1-D (n_features,)
        base_val  = float(expl.expected_value[cidx])

        # Manual waterfall bar chart
        order = np.argsort(np.abs(shap_vals))[::-1][:8]
        labels = [FEATURE_NAMES[i] for i in order]
        vals   = shap_vals[order]
        colors = ["#ef4444" if v>0 else "#22c55e" for v in vals]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1], alpha=0.85, height=0.6)
        ax.axvline(0, color="#1e293b", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xlabel("SHAP value (impact on high-risk probability)", fontsize=9)
        ax.set_title(f"Local SHAP — This Patient (risk class: {target_class})", fontsize=10, fontweight="500")
        ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.text(0.98, 0.02, f"Base value: {base_val:.3f}",
                transform=ax.transAxes, ha="right", fontsize=8, color="#64748b")
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION CURVES
# ──────────────────────────────────────────────────────────────────────────────
def plot_calibration(extras, cls="high"):
    if not extras or "cal_data" not in extras: return None
    data = extras["cal_data"].get(cls)
    if not data: return None

    fu = data["frac_uncal"]; mu = data["mean_uncal"]
    fc = data["frac_cal"];   mc = data["mean_cal"]
    bs_u = data["bs_uncal"]; bs_c = data["bs_cal"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
    ax.plot(mu, fu, "o-", color="#94a3b8", linewidth=1.8, markersize=5,
            label=f"Uncalibrated  (Brier={bs_u:.4f})")
    ax.plot(mc, fc, "s-", color=RISK_COLORS.get(cls,"#3b82f6"), linewidth=2, markersize=5,
            label=f"Calibrated  (Brier={bs_c:.4f})")
    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_ylabel("Fraction of positives", fontsize=9)
    ax.set_title(f"Reliability Diagram — class: {cls}", fontsize=10, fontweight="500")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_calibration_all(extras):
    """Side-by-side calibration curves for all 3 classes."""
    if not extras or "cal_data" not in extras: return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, cls in zip(axes, ["high","low","medium"]):
        data = extras["cal_data"].get(cls, {})
        if not data: continue
        fu=data["frac_uncal"]; mu=data["mean_uncal"]
        fc=data["frac_cal"];   mc=data["mean_cal"]
        bs_u=data["bs_uncal"]; bs_c=data["bs_cal"]
        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
        ax.plot(mu,fu,"o-",color="#94a3b8",lw=1.8,ms=5,label=f"Uncal Brier={bs_u:.4f}")
        ax.plot(mc,fc,"s-",color=RISK_COLORS.get(cls,"#3b82f6"),lw=2,ms=5,
                label=f"Cal Brier={bs_c:.4f}")
        ax.set_title(f"Class: {cls}", fontsize=10, fontweight="500")
        ax.set_xlabel("Mean pred. prob", fontsize=8)
        ax.set_ylabel("Fraction +", fontsize=8)
        ax.legend(fontsize=7.5); ax.tick_params(labelsize=7.5)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Probability Calibration — Reliability Diagrams (all 3 classes)",
                 fontsize=11, fontweight="500", y=1.02)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# LONGITUDINAL / TIME-SERIES SIMULATION
# ──────────────────────────────────────────────────────────────────────────────
def build_trajectory(base_inp, improvement_rate, models, extras):
    """
    Simulate 4 assessment timepoints assuming treatment is delivered.
    improvement_rate: 0.0 (no improvement) to 1.0 (fast response)
    Returns list of per-timepoint dicts.
    """
    if models is None: return []
    rf_clf = models["rf_clf"]; rf_reg = models["rf_reg"]
    cal    = extras.get("cal_pipeline") if extras else None

    # PHQ-9 decay: sigmoid-shaped reduction scaled by improvement_rate
    phq9_0 = base_inp["phq9"]
    phq9_trajectory = [
        phq9_0,
        max(10, round(phq9_0 - improvement_rate * phq9_0 * 0.15)),
        max(10, round(phq9_0 - improvement_rate * phq9_0 * 0.32)),
        max(10, round(phq9_0 - improvement_rate * phq9_0 * 0.50)),
    ]
    gad7_0 = base_inp["gad7"]
    gad7_trajectory = [
        gad7_0,
        max(0, round(gad7_0 - improvement_rate * gad7_0 * 0.12)),
        max(0, round(gad7_0 - improvement_rate * gad7_0 * 0.28)),
        max(0, round(gad7_0 - improvement_rate * gad7_0 * 0.45)),
    ]
    sleep_trajectory = [
        base_inp["sleep_quality"],
        max(1, base_inp["sleep_quality"] - round(improvement_rate * 0.8)),
        max(1, base_inp["sleep_quality"] - round(improvement_rate * 1.2)),
        max(1, base_inp["sleep_quality"] - round(improvement_rate * 1.5)),
    ]

    results = []
    for t_idx in range(4):
        tp = dict(base_inp)
        tp["phq9"]          = int(phq9_trajectory[t_idx])
        tp["gad7"]          = int(gad7_trajectory[t_idx])
        tp["sleep_quality"] = int(sleep_trajectory[t_idx])
        row = pd.DataFrame([{k:tp[k] for k in FEATURE_NAMES}])
        risk  = rf_clf.predict(row)[0]
        proba = dict(zip(rf_clf.classes_, rf_clf.predict_proba(row)[0].round(3)))
        scores = {t:round(float(s),1) for t,s in zip(TARGET_REG, rf_reg.predict(row)[0])}
        cal_p = None
        if cal:
            cal_p = dict(zip(cal.classes_, cal.predict_proba(row)[0].round(3)))
        results.append({
            "label":    TIMEPOINT_LABELS[t_idx],
            "phq9":     tp["phq9"],
            "gad7":     tp["gad7"],
            "sleep_q":  tp["sleep_quality"],
            "risk":     risk,
            "risk_proba": proba,
            "cal_proba":  cal_p,
            "scores":   scores,
        })
    return results


def plot_trajectory(traj):
    """Multi-panel trajectory chart: PHQ-9, risk probability, suitability scores."""
    if not traj: return None
    labels = [t["label"].split(" — ")[1] for t in traj]
    phq9s  = [t["phq9"] for t in traj]
    gad7s  = [t["gad7"] for t in traj]
    sleeps = [t["sleep_q"] for t in traj]
    high_p = [t["cal_proba"]["high"] if t["cal_proba"] else t["risk_proba"]["high"] for t in traj]
    med_s  = [t["scores"]["medication_first_suitability"]    for t in traj]
    psy_s  = [t["scores"]["psychotherapy_first_suitability"] for t in traj]
    comb_s = [t["scores"]["combined_care_suitability"]       for t in traj]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = range(4)

    # Panel 1 — Symptom trajectory
    ax1 = axes[0]
    ax1.plot(x, phq9s,  "o-", color="#3b82f6", lw=2, ms=6, label="PHQ-9")
    ax1.plot(x, gad7s,  "s-", color="#8b5cf6", lw=2, ms=6, label="GAD-7")
    ax1.plot(x, sleeps, "^-", color="#06b6d4", lw=1.5, ms=5, label="Sleep (1-5)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8, rotation=10)
    ax1.set_ylabel("Score", fontsize=9); ax1.legend(fontsize=8)
    ax1.set_title("Symptom Trajectory", fontsize=10, fontweight="500")
    ax1.spines[["top","right"]].set_visible(False); ax1.tick_params(labelsize=8)

    # Panel 2 — High-risk probability
    ax2 = axes[1]
    colors_bar = [RISK_COLORS[t["risk"]] for t in traj]
    bars = ax2.bar(x, high_p, color=colors_bar, alpha=0.8, width=0.5)
    ax2.axhline(0.5, color="#94a3b8", lw=1, linestyle="--", label="50% threshold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8, rotation=10)
    ax2.set_ylabel("P(high risk)", fontsize=9); ax2.set_ylim(0, 1)
    ax2.set_title("Calibrated High-Risk Probability", fontsize=10, fontweight="500")
    for bar, val in zip(bars, high_p):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.02,
                 f"{val:.0%}", ha="center", fontsize=8, fontweight="500")
    ax2.spines[["top","right"]].set_visible(False); ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8)

    # Panel 3 — Suitability scores
    ax3 = axes[2]
    ax3.plot(x, med_s,  "o-", color=SCORE_COLORS["medication_first_suitability"],   lw=2, ms=6, label="Medication")
    ax3.plot(x, psy_s,  "s-", color=SCORE_COLORS["psychotherapy_first_suitability"], lw=2, ms=6, label="Psychotherapy")
    ax3.plot(x, comb_s, "^-", color=SCORE_COLORS["combined_care_suitability"],       lw=2, ms=6, label="Combined Care")
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=8, rotation=10)
    ax3.set_ylabel("Suitability score (/100)", fontsize=9); ax3.set_ylim(0, 100)
    ax3.set_title("Suitability Score Trajectory", fontsize=10, fontweight="500")
    ax3.legend(fontsize=8)
    ax3.spines[["top","right"]].set_visible(False); ax3.tick_params(labelsize=8)

    fig.suptitle("Longitudinal Treatment Response Simulation", fontsize=11,
                 fontweight="500", y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# RENDERERS
# ──────────────────────────────────────────────────────────────────────────────
def render_score_card(label, score, color):
    pct=min(int(score),100)
    st.markdown(f"""
    <div class="score-card">
        <div class="lbl">{label}</div>
        <div class="val">{score:.0f}<span style="font-size:12px;color:#94a3b8;font-weight:400">/100</span></div>
        <div class="bo"><div class="bi" style="width:{pct}%;background:{color}"></div></div>
    </div>""", unsafe_allow_html=True)

def render_risk_badge(risk, proba=None, cal_proba=None):
    display_proba = cal_proba or proba
    if risk=="high":
        hp = display_proba.get("high",0) if display_proba else 0
        st.markdown(f"""
        <div class="risk-high-banner">
            <div class="rh-label">Early Non-Response Risk</div>
            <div class="rh-title">🔴 HIGH RISK</div>
            <div class="rh-sub">
                {'Calibrated' if cal_proba else 'Model'} confidence:
                <strong>{hp:.0%}</strong> — proactive escalation planning required.
            </div>
            <div class="rh-cl">
                <div style="font-size:10px;font-weight:700;color:#b91c1c;text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px;">Clinical Escalation Checklist</div>
                <div class="rh-check"><span>☐</span><span>Follow-up within 2 weeks of treatment initiation</span></div>
                <div class="rh-check"><span>☐</span><span>Consider combined care (pharmacotherapy + psychotherapy)</span></div>
                <div class="rh-check"><span>☐</span><span>Address sleep disturbance directly (consider CBT-I)</span></div>
                <div class="rh-check"><span>☐</span><span>If prior treatment failed, consult for switch/augmentation</span></div>
                <div class="rh-check"><span>☐</span><span>Assess safety and suicidality at each contact</span></div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        css="risk-medium" if risk=="medium" else "risk-low"
        icon="🟡" if risk=="medium" else "🟢"
        lbl="Medium Risk" if risk=="medium" else "Low Risk"
        st.markdown(f"""
        <div class="{css}">
            <div class="risk-label">Early Non-Response Risk</div>
            <div class="risk-value">{icon} {lbl}</div>
        </div>""", unsafe_allow_html=True)

def render_confidence_gauge(proba, cal_proba=None):
    dp = cal_proba or proba
    if not dp: return
    high_p=dp.get("high",0); med_p=dp.get("medium",0); low_p=dp.get("low",0)
    dominant=max(dp,key=dp.get)
    dcol={"high":"#dc2626","medium":"#d97706","low":"#16a34a"}[dominant]
    label_txt = "Calibrated probability" if cal_proba else "Raw model probability"
    st.markdown(f"""
    <div class="conf-wrap">
        <div class="conf-title">Model Confidence{' (calibrated)' if cal_proba else ''}</div>
        <div class="conf-row">
            <div class="conf-lbl" style="color:#dc2626">🔴 High</div>
            <div class="conf-bo"><div class="conf-bar-high" style="width:{high_p*100:.0f}%"></div></div>
            <div class="conf-pct">{high_p:.0%}</div>
        </div>
        <div class="conf-row">
            <div class="conf-lbl" style="color:#d97706">🟡 Medium</div>
            <div class="conf-bo"><div class="conf-bar-medium" style="width:{med_p*100:.0f}%"></div></div>
            <div class="conf-pct">{med_p:.0%}</div>
        </div>
        <div class="conf-row">
            <div class="conf-lbl" style="color:#16a34a">🟢 Low</div>
            <div class="conf-bo"><div class="conf-bar-low" style="width:{low_p*100:.0f}%"></div></div>
            <div class="conf-pct">{low_p:.0%}</div>
        </div>
        <div class="conf-foot">Predicted: <strong style="color:{dcol}">{dominant.upper()}</strong>
            &nbsp;·&nbsp; Confidence: <strong style="color:{dcol}">{dp.get(dominant,0):.0%}</strong>
            &nbsp;·&nbsp; <span style="color:#94a3b8">{label_txt}</span></div>
    </div>""", unsafe_allow_html=True)

def render_comparison_row(field, ml_val, rb_val):
    match=(ml_val.strip()==rb_val.strip())
    st.markdown(f"""
    <div class="cmp-row {'cmp-match' if match else 'cmp-differ'}">
        <div><div class="cmp-label">{field}</div><div class="cmp-val-ml">🤖 {ml_val}</div></div>
        <div style="text-align:right"><div class="cmp-label">Rule-Based</div>
            <div class="cmp-val-rb">{'✓' if match else '≠'} {rb_val}</div></div>
    </div>""", unsafe_allow_html=True)

def render_explanation(item):
    icons={"alert":"⚠️","warn":"📌","good":"✅","info":"ℹ️"}
    icon="🤖" if item.get("source")=="model" else icons.get(item["tone"],"ℹ️")
    css="model" if item.get("source")=="model" else item["tone"]
    st.markdown(f"""
    <div class="explain-item {css}">
        <span style="flex-shrink:0">{icon}</span><span>{item['text']}</span>
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🧠 Gene2Care-AI")
st.sidebar.markdown('<div class="scenario-label">⚡ Demo Scenarios</div>',unsafe_allow_html=True)
for name, data in DEMO_SCENARIOS.items():
    if st.sidebar.button(name, use_container_width=True, help=data["desc"]):
        for k,v in data.items():
            if k!="desc": st.session_state[f"inp_{k}"]=v
        st.session_state["run_demo"]=True
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Manual Input")

def ss(k,d): return st.session_state.get(f"inp_{k}",d)

age    = st.sidebar.slider("Age",16,40,ss("age",22))
gender = st.sidebar.selectbox("Gender",["Female","Male","Other"],
             index=["Female","Male","Other"].index(ss("gender","Female")))
phq9   = st.sidebar.slider("PHQ-9",10,27,ss("phq9",16),help="10–27 = moderate to severe")
gad7   = st.sidebar.slider("GAD-7",0,21,ss("gad7",10))
sleep_quality    = st.sidebar.slider("Sleep Quality",1,5,ss("sleep_quality",3),help="1=good · 5=poor")
symptom_duration = st.sidebar.slider("Duration (months)",2,72,ss("symptom_duration_months",6))
recurrence = st.sidebar.slider("Recurrence",0,5,ss("recurrence_count",0))
ptx_opts=["none","medication_only","psychotherapy_only","both","failed_or_discontinued"]
prior_tx=st.sidebar.selectbox("Prior Treatment",ptx_opts,
             index=ptx_opts.index(ss("prior_treatment_status","none")))
fi_opts=["mild","moderate","severe"]
func_impairment=st.sidebar.selectbox("Functional Impairment",fi_opts,
                    index=fi_opts.index(ss("functional_impairment","mild")))
ta_opts=["low","medium","high"]
therapy_access=st.sidebar.selectbox("Therapy Access",ta_opts,
                   index=ta_opts.index(ss("therapy_access","high")))

st.sidebar.markdown("---")
run_btn=st.sidebar.button("⚡ Run Assessment",use_container_width=True,type="primary")
run_btn=run_btn or st.session_state.pop("run_demo",False)

inputs={
    "age":age,"gender":gender,"phq9":phq9,"gad7":gad7,
    "sleep_quality":sleep_quality,"symptom_duration_months":symptom_duration,
    "recurrence_count":recurrence,"prior_treatment_status":prior_tx,
    "functional_impairment":func_impairment,"therapy_access":therapy_access,
}

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <span style="font-size:33px">🧠</span>
  <div>
    <h1>Gene2Care-AI</h1>
    <p>Clinical Decision Support — Treatment Stratification &amp; Longitudinal Response Modeling</p>
    <span class="badge">Synthetic prototype · Research stage only · Not for clinical use</span>
  </div>
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if not run_btn:
    cl,cr=st.columns([2,1])
    with cl:
        st.markdown("""
**v3 capabilities:**
- 🤖 ML prediction with calibrated probabilities
- ⚖️ DSM rule-based baseline comparison
- 🔍 Dual-source explanation (clinical logic + model patterns)
- 📊 **SHAP Explainability** — global feature importance + local patient explanation
- 🎯 **Probability Calibration** — reliability diagrams, Brier scores (3 classes)
- 📈 **Longitudinal Modeling** — treatment response trajectory simulation (T0→T3)
        """)
        st.info("👈  Select a **demo scenario** or adjust inputs, then click **Run Assessment**.")
    with cr:
        for name, data in DEMO_SCENARIOS.items():
            st.markdown(f"**{name}** — _{data['desc']}_")
else:
    rb_result = rule_based_predict(inputs)
    ml_result = ml_predict(inputs, models, extras)
    using_fb  = (ml_result is None)
    if using_fb:
        ml_result={**rb_result,"scores":{t:50.0 for t in TARGET_REG},"risk_proba":None,"cal_proba":None}

    tabs = st.tabs([
        "🤖 Prediction",
        "⚖️ Comparison",
        "🔍 Explanation",
        "📊 SHAP",
        "🎯 Calibration",
        "📈 Longitudinal",
    ])

    # ══════ TAB 1 — PREDICTION ══════
    with tabs[0]:
        if using_fb:
            st.error("**gene2care_models.joblib not found.** Showing rule-based fallback.")
        st.markdown('<div class="sec">Suitability Scores</div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1: render_score_card("Medication First",ml_result["scores"]["medication_first_suitability"],SCORE_COLORS["medication_first_suitability"])
        with c2: render_score_card("Psychotherapy First",ml_result["scores"]["psychotherapy_first_suitability"],SCORE_COLORS["psychotherapy_first_suitability"])
        with c3: render_score_card("Combined Care",ml_result["scores"]["combined_care_suitability"],SCORE_COLORS["combined_care_suitability"])

        st.markdown('<div class="sec">Risk Assessment &amp; Confidence</div>',unsafe_allow_html=True)
        cr1,cr2=st.columns(2)
        with cr1: render_risk_badge(ml_result["risk"],ml_result.get("risk_proba"),ml_result.get("cal_proba"))
        with cr2:
            if not using_fb:
                render_confidence_gauge(ml_result.get("risk_proba"),ml_result.get("cal_proba"))

        st.markdown('<div class="sec">Recommendations</div>',unsafe_allow_html=True)
        r1,r2,r3=st.columns(3)
        with r1: st.markdown(f"**💊 Medication**\n\n{ml_result['medication']}")
        with r2: st.markdown(f"**🗣️ Therapy**\n\n{ml_result['therapy']}")
        with r3: st.markdown(f"**🤝 Combined Care**\n\n{ml_result['combined_care']}")

    # ══════ TAB 2 — COMPARISON ══════
    with tabs[1]:
        if using_fb:
            st.warning("ML model unavailable — comparison not possible.")
        else:
            fields={"Medication":(ml_result["medication"],rb_result["medication"]),
                    "Therapy":(ml_result["therapy"],rb_result["therapy"]),
                    "Combined Care":(ml_result["combined_care"],rb_result["combined_care"]),
                    "Risk Level":(ml_result["risk"],rb_result["risk"])}
            nd=sum(1 for mv,rv in fields.values() if mv.strip()!=rv.strip())
            if nd==0:
                st.markdown('<div class="model-agrees">✅ <strong>Model agrees</strong> with rule-based baseline on all 4 outputs.</div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="model-differs">⚠️ <strong>Model differs</strong> on {nd} output{"s" if nd>1 else ""} — ML may capture interaction effects the rule system misses.</div>',unsafe_allow_html=True)
            st.markdown('<div class="sec">Output Comparison</div>',unsafe_allow_html=True)
            for f,(mv,rv) in fields.items(): render_comparison_row(f,mv,rv)
            st.markdown('<div class="sec">Score Table</div>',unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"Treatment Path":[SCORE_LABELS[t] for t in TARGET_REG],
                "ML Score (/100)":[f"{ml_result['scores'][t]:.0f}" for t in TARGET_REG]
                }).set_index("Treatment Path"),use_container_width=True)
            st.caption("Rule-based baseline does not produce numeric suitability scores.")

    # ══════ TAB 3 — EXPLANATION ══════
    with tabs[2]:
        all_items=build_explanation(inputs, ml_result if not using_fb else None)
        clinical=[i for i in all_items if i.get("source")=="clinical"]
        model_i =[i for i in all_items if i.get("source")=="model"]
        st.markdown('<div class="sec">Clinical Logic (DSM-Inspired)</div>',unsafe_allow_html=True)
        st.caption("Grounded in validated clinical criteria and treatment guidelines.")
        for item in clinical: render_explanation(item)
        if model_i and not using_fb:
            st.markdown('<div class="sec">Model Pattern Explanations 🤖</div>',unsafe_allow_html=True)
            st.caption("Why the model produced this specific score/risk pattern — referencing learned data relationships. Purple border = model-derived.")
            for item in model_i: render_explanation(item)
        st.markdown('<div class="sec">Input Summary</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"PHQ-9":phq9,"GAD-7":gad7,"Sleep":f"{sleep_quality}/5",
            "Duration":f"{symptom_duration}mo","Recurrence":recurrence,"Prior Tx":prior_tx,
            "Impairment":func_impairment,"Access":therapy_access}
        ]).T.rename(columns={0:"Value"}),use_container_width=True)

    # ══════ TAB 4 — SHAP ══════
    with tabs[3]:
        if using_fb or not extras:
            st.warning("SHAP tab requires gene2care_extras.joblib and gene2care_models.joblib.")
        else:
            shap_avail = extras.get("shap_available", False)
            if shap_avail:
                st.success("SHAP values available (TreeExplainer, precomputed on 200 test samples).")
            else:
                st.info("SHAP not installed — showing RF Gini importance as fallback. "
                        "Install with `pip install shap` for full SHAP explanations.")

            st.markdown('<div class="sec">Global Feature Importance</div>',unsafe_allow_html=True)
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                risk_class = st.selectbox("Risk class", ["high","low","medium"],
                                          key="shap_cls", index=0)
                fig = plot_global_importance(extras, target_class=risk_class)
                if fig: show_fig(fig, f"Top features for predicting risk = {risk_class}")
                else:   st.caption("No importance data available.")
            with gcol2:
                reg_target = st.selectbox("Regression target", TARGET_REG,
                                          format_func=lambda x: SCORE_LABELS[x],
                                          key="shap_reg")
                fig = plot_global_importance(extras, regression_target=reg_target)
                if fig: show_fig(fig, f"Top features for {SCORE_LABELS[reg_target]}")
                else:   st.caption("No regression importance data available.")

            st.markdown('<div class="sec">Local Explanation — This Patient</div>',unsafe_allow_html=True)
            if shap_avail:
                local_cls = st.selectbox("Explain for class", ["high","low","medium"],
                                         key="local_cls")
                fig = plot_local_shap(inputs, extras, target_class=local_cls)
                if fig:
                    show_fig(fig, "Red bars = features pushing toward this class. "
                             "Green bars = features pushing away.")
                else:
                    st.caption("Local SHAP plot unavailable for this input.")
            else:
                st.caption("Local SHAP requires `pip install shap`.")
                # Show a table of RF feature importances as fallback
                rf = extras.get("rf_raw")
                if rf:
                    imp_df = pd.DataFrame({
                        "Feature":    FEATURE_NAMES,
                        "Importance": rf.feature_importances_.round(4),
                    }).sort_values("Importance", ascending=False).reset_index(drop=True)
                    st.dataframe(imp_df, use_container_width=True)

    # ══════ TAB 5 — CALIBRATION ══════
    with tabs[4]:
        if using_fb or not extras:
            st.warning("Calibration tab requires gene2care_extras.joblib.")
        else:
            st.markdown('<div class="sec">What is probability calibration?</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "A perfectly calibrated model means: when it predicts 70% probability of high risk, "
                "70% of those patients actually have high risk. The **reliability diagram** plots "
                "predicted probability (x) vs observed fraction (y). The **Brier score** measures "
                "mean squared error of probability predictions — lower is better.")

            st.markdown('<div class="sec">Brier Score Comparison</div>',unsafe_allow_html=True)
            bcols = st.columns(3)
            for col, cls in zip(bcols, ["high","low","medium"]):
                cd = extras["cal_data"].get(cls,{})
                if cd:
                    delta = cd["bs_uncal"] - cd["bs_cal"]
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="mlbl">Class: {cls}</div>
                            <div class="mval">{cd['bs_cal']:.4f}</div>
                            <div class="msub">Calibrated Brier</div>
                            <div class="msub" style="color:#16a34a">▼ {delta:.4f} vs uncal</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec">Reliability Diagrams — All 3 Classes</div>',
                        unsafe_allow_html=True)
            fig = plot_calibration_all(extras)
            if fig:
                show_fig(fig, "Isotonic calibration (5-fold CV). Closer to the diagonal = better calibrated.")

            st.markdown('<div class="sec">Single Class Detail</div>',unsafe_allow_html=True)
            detail_cls = st.selectbox("Select class", ["high","low","medium"],
                                      key="cal_cls")
            fig2 = plot_calibration(extras, cls=detail_cls)
            if fig2:
                show_fig(fig2)

    # ══════ TAB 6 — LONGITUDINAL ══════
    with tabs[5]:
        if using_fb or not extras:
            st.warning("Longitudinal tab requires gene2care_models.joblib and gene2care_extras.joblib.")
        else:
            st.markdown('<div class="sec">Treatment Response Trajectory Simulation</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "Simulates how predicted risk and suitability scores change across "
                "4 assessment timepoints (Baseline → Week 2 → Week 4 → Week 8) "
                "as symptoms improve under treatment. Adjust the **improvement rate** "
                "to model fast vs slow responders.")

            rate = st.slider(
                "Improvement rate",
                min_value=0.0, max_value=1.0, value=0.6, step=0.1,
                help="0 = no improvement  ·  1.0 = fast response (50% PHQ-9 reduction by week 8)")

            traj = build_trajectory(inputs, rate, models, extras)

            if traj:
                # Summary table
                st.markdown('<div class="sec">Assessment Timepoints</div>',
                            unsafe_allow_html=True)
                rows = []
                for t in traj:
                    hp = t["cal_proba"]["high"] if t["cal_proba"] else t["risk_proba"]["high"]
                    rows.append({
                        "Timepoint":     t["label"],
                        "PHQ-9":         t["phq9"],
                        "GAD-7":         t["gad7"],
                        "Sleep":         f"{t['sleep_q']}/5",
                        "Risk":          t["risk"].upper(),
                        "P(high)":       f"{hp:.0%}",
                        "Med /100":      f"{t['scores']['medication_first_suitability']:.0f}",
                        "Psy /100":      f"{t['scores']['psychotherapy_first_suitability']:.0f}",
                        "Comb /100":     f"{t['scores']['combined_care_suitability']:.0f}",
                    })
                traj_df = pd.DataFrame(rows).set_index("Timepoint")
                st.dataframe(traj_df, use_container_width=True)

                # Trajectory chart
                st.markdown('<div class="sec">Trajectory Chart</div>',unsafe_allow_html=True)
                fig = plot_trajectory(traj)
                if fig:
                    show_fig(fig,
                             "Left: symptom scores over time. "
                             "Centre: calibrated high-risk probability (bar colour = predicted class). "
                             "Right: suitability scores — note how psychotherapy suitability "
                             "may increase as severity declines.")

                # Clinical interpretation
                st.markdown('<div class="sec">Interpretation</div>',unsafe_allow_html=True)
                t0_risk = traj[0]["risk"]; t3_risk = traj[-1]["risk"]
                t0_hp   = (traj[0]["cal_proba"] or traj[0]["risk_proba"]).get("high",0)
                t3_hp   = (traj[-1]["cal_proba"] or traj[-1]["risk_proba"]).get("high",0)
                delta   = t0_hp - t3_hp

                if delta >= 0.3:
                    st.success(f"✅ Substantial risk reduction: P(high) fell from "
                               f"{t0_hp:.0%} → {t3_hp:.0%} by Week 8 at rate={rate:.1f}. "
                               f"Treatment response signal is strong.")
                elif delta >= 0.1:
                    st.warning(f"📌 Moderate risk reduction: P(high) {t0_hp:.0%} → {t3_hp:.0%}. "
                               f"Monitor closely for early non-response.")
                else:
                    st.error(f"⚠️ Minimal risk change at this improvement rate: "
                             f"{t0_hp:.0%} → {t3_hp:.0%}. "
                             f"This trajectory suggests high non-response risk — "
                             f"combined care escalation should be considered early.")

                if traj[-1]["scores"]["psychotherapy_first_suitability"] > \
                   traj[0]["scores"]["psychotherapy_first_suitability"] + 5:
                    st.info("ℹ️ Psychotherapy suitability increases as severity decreases — "
                            "transitioning from combined to psychotherapy-only may be appropriate "
                            "at Week 8 if clinical response is confirmed.")

# ──────────────────────────────────────────────────────────────────────────────
# DISCLAIMER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Research Prototype Only.</strong> This application uses a synthetic dataset
    and has not been clinically validated. All outputs are illustrative and must not be used
    to inform real clinical decisions.
</div>""", unsafe_allow_html=True)
