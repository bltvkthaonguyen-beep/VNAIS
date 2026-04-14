from core.rules.rule_engine import rule_based_predict
from core.ml.predictor import ml_predict
from core.ml.loader import load_models
from core.explain.explanation import build_explanation
from config.scenarios import DEMO_SCENARIOS


def test_all_demo_scenarios_run():
    models = load_models()

    for name, scenario in DEMO_SCENARIOS.items():
        inputs = {k: v for k, v in scenario.items() if k != "desc"}

        rb = rule_based_predict(inputs)
        ml = ml_predict(inputs, models)

        assert "risk" in rb

        if ml is not None:
            assert "scores" in ml


def test_pipeline_end_to_end(sample_input):
    models = load_models()

    rb = rule_based_predict(sample_input)
    ml = ml_predict(sample_input, models)

    explanation = build_explanation(sample_input, ml)

    assert rb is not None
    assert isinstance(explanation, list)
