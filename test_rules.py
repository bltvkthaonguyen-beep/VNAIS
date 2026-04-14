from core.rules.rule_engine import rule_based_predict


def test_rule_based_returns_keys(sample_input):
    result = rule_based_predict(sample_input)

    assert "medication" in result
    assert "therapy" in result
    assert "combined_care" in result
    assert "risk" in result


def test_rule_based_high_risk(high_risk_input):
    result = rule_based_predict(high_risk_input)

    assert result["risk"] == "high"


def test_rule_based_low_vs_high(sample_input, high_risk_input):
    low_result = rule_based_predict(sample_input)
    high_result = rule_based_predict(high_risk_input)

    assert low_result["risk"] != high_result["risk"]
