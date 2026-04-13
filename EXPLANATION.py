# ──────────────────────────────────────────────────────────────────────────────
# EXPLANATION 
# ──────────────────────────────────────────────────────────────────────────────
def build_explanation(inputs, ml_result):
    """
    Returns items tagged source='clinical' or source='model'.
    Clinical items are grounded in DSM heuristics.
    Model items reference actual predicted scores and explain
    why the model produced that pattern from training data.
    """
    items = []
    phq9 = inputs["phq9"]; gad7 = inputs["gad7"]
    sleep_q  = inputs["sleep_quality"]
    duration = inputs["symptom_duration_months"]
    relapse  = inputs["recurrence_count"]
    fi       = inputs["functional_impairment"]
    prior_tx = inputs["prior_treatment_status"]
    ta       = inputs["therapy_access"]
 
    has_scores = ml_result is not None and "scores" in ml_result
    med_s  = ml_result["scores"]["medication_first_suitability"]    if has_scores else None
    psy_s  = ml_result["scores"]["psychotherapy_first_suitability"] if has_scores else None
    comb_s = ml_result["scores"]["combined_care_suitability"]       if has_scores else None
    risk   = ml_result.get("risk")                                  if ml_result else None
    proba  = ml_result.get("risk_proba", {})                        if ml_result else {}
 
    # ── CLINICAL ITEMS ────────────────────────────────────────
    if phq9 >= 20:
        items.append({"source":"clinical","tone":"alert",
            "text":f"PHQ-9 = {phq9} (severe range) — DSM criteria at this level indicate "
                   f"pharmacotherapy and close monitoring. Severity is a primary driver of "
                   f"treatment intensity and non-response risk."})
    elif phq9 >= 15:
        items.append({"source":"clinical","tone":"warn",
            "text":f"PHQ-9 = {phq9} (moderately severe) — structured treatment is strongly "
                   f"recommended. Both pharmacotherapy and psychotherapy are clinically appropriate."})
    else:
        items.append({"source":"clinical","tone":"good",
            "text":f"PHQ-9 = {phq9} (moderate range) — psychotherapy-first approaches are "
                   f"viable before escalating to medication."})
 
    if gad7 >= 15:
        items.append({"source":"clinical","tone":"alert",
            "text":f"GAD-7 = {gad7} (severe anxiety) — high comorbid anxiety independently "
                   f"elevates non-response risk and indicates anxiety-focused therapy "
                   f"or combined care."})
    elif gad7 >= 10:
        items.append({"source":"clinical","tone":"warn",
            "text":f"GAD-7 = {gad7} (moderate anxiety comorbidity) — anxious depression "
                   f"typically responds better to combined treatment."})
 
    if sleep_q >= 4:
        items.append({"source":"clinical","tone":"alert",
            "text":f"Sleep quality = {sleep_q}/5 (very poor) — sleep disturbance is both "
                   f"a symptom amplifier and a validated predictor of non-response. "
                   f"CBT-I alongside main treatment may improve outcomes."})
    elif sleep_q <= 2:
        items.append({"source":"clinical","tone":"good",
            "text":f"Sleep quality = {sleep_q}/5 (good) — intact sleep is a protective "
                   f"factor associated with better treatment response."})
 
    if duration > 24:
        items.append({"source":"clinical","tone":"alert",
            "text":f"Symptom duration = {duration} months (chronic, >2 years) — chronicity "
                   f"reduces single-modality response rates. Combined care and maintenance "
                   f"planning are clinically indicated."})
    elif duration > 12:
        items.append({"source":"clinical","tone":"warn",
            "text":f"Symptom duration = {duration} months — sub-chronic; treatment history "
                   f"becomes important for sequencing decisions."})
 
    if relapse >= 3:
        items.append({"source":"clinical","tone":"alert",
            "text":f"{relapse} prior episodes — high recurrence is a strong predictor of "
                   f"future episodes and poor response. Long-term maintenance and "
                   f"relapse-prevention CBT should be planned."})
    elif relapse >= 1:
        items.append({"source":"clinical","tone":"warn",
            "text":f"{relapse} prior episode(s) — recurrence shifts the profile toward "
                   f"combined or stepped-care approaches."})
 
    if prior_tx == "failed_or_discontinued":
        items.append({"source":"clinical","tone":"alert",
            "text":"Prior treatment failed or was discontinued — first-line SSRI is no "
                   "longer appropriate. Consider switching to a different class (SNRI, "
                   "atypical antidepressant) or augmentation strategy."})
    elif prior_tx == "none":
        items.append({"source":"clinical","tone":"good",
            "text":"No prior treatment history — standard first-line approach is appropriate."})
 
    if fi == "severe":
        items.append({"source":"clinical","tone":"alert",
            "text":"Severe functional impairment — the illness is substantially affecting "
                   "daily functioning. Combined care and more intensive monitoring are "
                   "strongly indicated."})
    elif fi == "mild":
        items.append({"source":"clinical","tone":"good",
            "text":"Mild functional impairment — preserved functioning supports "
                   "psychotherapy-first as a reasonable starting point."})
 
    if ta == "low":
        items.append({"source":"clinical","tone":"warn",
            "text":"Low therapy access — structural barriers make in-person psychotherapy "
                   "difficult. Digital CBT or pharmacotherapy may be the most practical "
                   "first step regardless of clinical optimality."})
    elif ta == "high":
        items.append({"source":"clinical","tone":"good",
            "text":"High therapy access — structured psychotherapy (CBT, BA) is feasible "
                   "and should be offered."})
 
    # ── MODEL SCORE PATTERN ITEMS ─────────────────────────────
    if has_scores:
        # Dominant score explanation
        top_t = max(ml_result["scores"], key=ml_result["scores"].get)
        top_s = ml_result["scores"][top_t]
        items.append({"source":"model","tone":"info",
            "text":f"Highest model suitability: {SCORE_LABELS[top_t]} = {top_s:.0f}/100. "
                   f"The model assigned this based on the joint weight of PHQ-9, GAD-7, "
                   f"sleep quality, chronicity, and therapy access learned from training data."})
 
        # Combined care score driven by multi-factor burden
        if comb_s >= 65:
            drivers = []
            if phq9 >= 18:   drivers.append(f"PHQ-9={phq9}")
            if gad7 >= 12:   drivers.append(f"GAD-7={gad7}")
            if sleep_q >= 4: drivers.append(f"poor sleep ({sleep_q}/5)")
            if duration>12:  drivers.append(f"duration {duration}mo")
            if relapse >= 2: drivers.append(f"{relapse} relapses")
            if drivers:
                items.append({"source":"model","tone":"warn",
                    "text":f"Combined care score = {comb_s:.0f}/100, elevated by: "
                           f"{', '.join(drivers)}. The model learned that this combination "
                           f"predicts higher benefit from combined treatment."})
 
        # Psychotherapy suppressed despite high access
        if psy_s < 45 and ta == "high":
            items.append({"source":"model","tone":"info",
                "text":f"Psychotherapy score = {psy_s:.0f}/100 despite high access. "
                       f"The model learned that high severity (PHQ-9={phq9}), chronicity "
                       f"({duration}mo), or recurrence ({relapse}) can suppress the "
                       f"psychotherapy signal even when structural access is not a barrier."})
 
        # Medication elevated by low access (access-constraint artifact)
        if med_s >= 55 and ta == "low":
            items.append({"source":"model","tone":"warn",
                "text":f"Medication score = {med_s:.0f}/100 is partly elevated by low therapy "
                       f"access — the model captures the real-world pattern that patients "
                       f"with access barriers are more likely to receive pharmacotherapy. "
                       f"This is a structural artifact, not pure clinical optimality."})
 
        # Risk confidence explanation
        if risk == "high" and proba.get("high", 0) >= 0.6:
            items.append({"source":"model","tone":"alert",
                "text":f"High-risk confidence = {proba['high']:.0%}. The model assigns strong "
                       f"confidence based on the co-occurrence of: "
                       f"{'chronic duration, ' if duration>12 else ''}"
                       f"{'multiple relapses, ' if relapse>=2 else ''}"
                       f"{'failed prior treatment, ' if prior_tx=='failed_or_discontinued' else ''}"
                       f"{'poor sleep, ' if sleep_q>=4 else ''}"
                       f"high severity. This pattern was strongly associated with non-response "
                       f"in the training data."})
        elif risk == "low" and proba.get("low", 0) >= 0.7:
            items.append({"source":"model","tone":"good",
                "text":f"Low-risk confidence = {proba['low']:.0%}. The model assigns high "
                       f"confidence — the absence of chronicity, recurrence, treatment failure, "
                       f"and good sleep creates a clear low-risk signal."})
        elif ml_result and proba.get(risk, 0) < 0.50:
            items.append({"source":"model","tone":"warn",
                "text":f"Risk = {(risk or '').upper()} but model confidence is only "
                       f"{proba.get(risk,0):.0%}. This profile sits near a decision boundary "
                       f"— clinical judgement should carry extra weight."})
 
    return items
