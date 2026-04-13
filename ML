# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    try:
        return joblib.load("gene2care_models.joblib")
    except FileNotFoundError:
        return None
 
models = load_models()

# ──────────────────────────────────────────────────────────────────────────────
# ML PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def ml_predict(inputs, models):
    if models is None: return None
    row    = pd.DataFrame([{k: inputs[k] for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES}])
    rf_clf = models["rf_clf"]; rf_reg = models["rf_reg"]
    risk_label = rf_clf.predict(row)[0]
    risk_proba = dict(zip(rf_clf.classes_, rf_clf.predict_proba(row)[0].round(3)))
    scores = {t: round(float(s), 1) for t, s in zip(TARGET_REG, rf_reg.predict(row)[0])}
 
    med_s = scores["medication_first_suitability"]
    psy_s = scores["psychotherapy_first_suitability"]
    comb_s = scores["combined_care_suitability"]
 
    if inputs["prior_treatment_status"] == "failed_or_discontinued": med_label = "🔄 Switch / Augment"
    elif med_s >= 65: med_label = "💊 SSRI First-Line"
    elif med_s >= 45: med_label = "💊 SSRI or SNRI"
    else:             med_label = "⏳ Not Primary"
 
    if psy_s >= 65:
        ther_label = "📱 Digital / Self-Guided CBT" if inputs["therapy_access"]=="low" \
                     else ("🎯 CBT with Anxiety Focus" if inputs["gad7"]>=15 else "🗣️ Standard CBT")
    elif psy_s >= 45: ther_label = "🔁 CBT + Behavioural Activation"
    else:             ther_label = "🌱 Low-Intensity CBT"
 
    comb_label = "✅ Combined Care Indicated" if comb_s >= 65 else "➖ Single Modality"
 
    return {"medication":med_label,"therapy":ther_label,"combined_care":comb_label,
            "risk":risk_label,"risk_proba":risk_proba,"scores":scores}
