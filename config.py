# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
NUMERIC_FEATURES     = ["age", "phq9", "gad7", "sleep_quality",
                         "symptom_duration_months", "recurrence_count"]
CATEGORICAL_FEATURES = ["gender", "prior_treatment_status",
                         "functional_impairment", "therapy_access"]
TARGET_REG = ["medication_first_suitability",
              "psychotherapy_first_suitability",
              "combined_care_suitability"]
TARGET_CLF = "early_nonresponse_risk"
 
SCORE_LABELS = {
    "medication_first_suitability":    "Medication First",
    "psychotherapy_first_suitability": "Psychotherapy First",
    "combined_care_suitability":       "Combined Care",
}
SCORE_COLORS = {
    "medication_first_suitability":    "#3b82f6",
    "psychotherapy_first_suitability": "#8b5cf6",
    "combined_care_suitability":       "#06b6d4",
}
 
# ──────────────────────────────────────────────────────────────────────────────
# DEMO SCENARIOS  (pre-verified ML outputs noted in comments)
# ──────────────────────────────────────────────────────────────────────────────
DEMO_SCENARIOS = {
    "🌱 First Episode": {
        "desc": "Young adult, first episode, moderate depression, high therapy access",
        "age": 20, "gender": "Female", "phq9": 14, "gad7": 8, "sleep_quality": 2,
        "symptom_duration_months": 4, "recurrence_count": 0,
        "prior_treatment_status": "none", "functional_impairment": "mild",
        "therapy_access": "high",
        # ML: risk=low (86%), psy=82, med=49, comb=52
    },
    "😰 Severe + Anxious": {
        "desc": "Severe depression, high anxiety, poor sleep — medium-high risk",
        "age": 27, "gender": "Female", "phq9": 23, "gad7": 18, "sleep_quality": 5,
        "symptom_duration_months": 10, "recurrence_count": 1,
        "prior_treatment_status": "medication_only", "functional_impairment": "severe",
        "therapy_access": "medium",
        # ML: risk=medium (65%), comb=79, med=61, psy=49
    },
    "🔁 Recurrent Chronic": {
        "desc": "Chronic, 3 relapses, failed prior treatment — HIGH non-response risk",
        "age": 36, "gender": "Male", "phq9": 19, "gad7": 14, "sleep_quality": 4,
        "symptom_duration_months": 36, "recurrence_count": 3,
        "prior_treatment_status": "failed_or_discontinued", "functional_impairment": "moderate",
        "therapy_access": "low",
        # ML: risk=high (81%), med=71, comb=75, psy=32
    },
    "🚧 Low Access": {
        "desc": "Moderate depression, no therapy access — access-driven medication uplift",
        "age": 29, "gender": "Male", "phq9": 17, "gad7": 11, "sleep_quality": 4,
        "symptom_duration_months": 9, "recurrence_count": 1,
        "prior_treatment_status": "none", "functional_impairment": "moderate",
        "therapy_access": "low",
        # ML: risk=medium (54%), med=61, psy=50, comb=60
    },
}
