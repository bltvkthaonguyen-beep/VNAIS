import pytest

@pytest.fixture
def sample_input():
    return {
        "age": 25,
        "gender": "Female",
        "phq9": 18,
        "gad7": 12,
        "sleep_quality": 4,
        "symptom_duration_months": 12,
        "recurrence_count": 1,
        "prior_treatment_status": "none",
        "functional_impairment": "moderate",
        "therapy_access": "medium",
    }


@pytest.fixture
def high_risk_input():
    return {
        "age": 35,
        "gender": "Male",
        "phq9": 24,
        "gad7": 18,
        "sleep_quality": 5,
        "symptom_duration_months": 36,
        "recurrence_count": 3,
        "prior_treatment_status": "failed_or_discontinued",
        "functional_impairment": "severe",
        "therapy_access": "low",
    }
