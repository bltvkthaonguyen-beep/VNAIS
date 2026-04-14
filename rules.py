# ──────────────────────────────────────────────────────────────────────────────
# RULE-BASED BASELINE
# ──────────────────────────────────────────────────────────────────────────────
def get_phq9_band(phq9):
    if phq9 <= 9:  return "mild"
    if phq9 <= 14: return "moderate"
    if phq9 <= 19: return "moderately_severe"
    return "severe"
 
def get_gad7_band(gad7):
    if gad7 <= 4:  return "minimal"
    if gad7 <= 9:  return "mild"
    if gad7 <= 14: return "moderate"
    return "severe"
 
def rule_based_predict(inputs):
    phq9 = inputs["phq9"]; gad7 = inputs["gad7"]
    sleep_q  = inputs["sleep_quality"]
    duration = inputs["symptom_duration_months"]
    relapse  = inputs["recurrence_count"]
    fi       = inputs["functional_impairment"]
    ta       = inputs["therapy_access"]
    prior_tx = inputs["prior_treatment_status"]
    phq_band = get_phq9_band(phq9); gad_band = get_gad7_band(gad7)
    fi_severe = (fi == "severe"); failed = (prior_tx == "failed_or_discontinued")
    chronic   = (duration > 12)
 
    if failed:                                       medication = "🔄 Switch / Augment"
    elif phq_band in ("moderately_severe","severe"): medication = "💊 SSRI First-Line"
    elif relapse >= 1 and phq9 > 14:                 medication = "💊 SSRI or SNRI"
    else:                                            medication = "⏳ Not Primary"
 
    if ta == "low":                                  therapy = "📱 Digital / Self-Guided CBT"
    elif phq_band=="mild" and gad_band in ("minimal","mild"): therapy = "🌱 Low-Intensity CBT"
    elif gad_band in ("moderate","severe"):          therapy = "🎯 CBT with Anxiety Focus"
    elif chronic:                                    therapy = "🔁 CBT + Behavioural Activation"
    else:                                            therapy = "🗣️ Standard CBT"
 
    sev = (phq9/27)*4 + (gad7/21)*2 + (sleep_q-1)/4
    combined = "✅ Combined Care Indicated" if (sev>5 or relapse>=2 or fi_severe) else "➖ Single Modality"
 
    risk_pts = 0
    if chronic:    risk_pts += 1
    if relapse>=1: risk_pts += 1
    if failed:     risk_pts += 2
    if sleep_q>=4: risk_pts += 1
    if fi_severe:  risk_pts += 1
    if phq9>=20:   risk_pts += 1
    risk = "high" if risk_pts>=4 else ("medium" if risk_pts>=2 else "low")
 
    return {"medication":medication,"therapy":therapy,"combined_care":combined,
            "risk":risk,"risk_pts":risk_pts}
