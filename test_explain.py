from core.explain.explanation import build_explanation


def test_explanation_not_empty(sample_input):
    items = build_explanation(sample_input, None)

    assert isinstance(items, list)
    assert len(items) > 0


def test_explanation_contains_clinical(sample_input):
    items = build_explanation(sample_input, None)

    has_clinical = any(i.get("source") == "clinical" for i in items)
    assert has_clinical


def test_explanation_with_ml(sample_input):
    fake_ml_result = {
        "risk": "medium",
        "scores": {
            "medication_first_suitability": 60,
            "psychotherapy_first_suitability": 50,
            "combined_care_suitability": 70,
        },
        "risk_proba": {"low": 0.2, "medium": 0.5, "high": 0.3},
    }

    items = build_explanation(sample_input, fake_ml_result)

    has_model = any(i.get("source") == "model" for i in items)
    assert has_model
