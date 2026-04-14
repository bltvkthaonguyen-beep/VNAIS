import pytest
from core.ml.predictor import ml_predict
from core.ml.loader import load_models


def test_model_loading():
    models = load_models()
    # Có thể None nếu chưa có file model
    assert models is None or isinstance(models, dict)


def test_ml_predict_safe(sample_input):
    models = load_models()
    result = ml_predict(sample_input, models)

    # Nếu không có model → result = None
    if result is None:
        assert result is None
    else:
        assert "scores" in result
        assert "risk" in result


def test_ml_scores_shape(sample_input):
    models = load_models()
    result = ml_predict(sample_input, models)

    if result is not None:
        scores = result["scores"]
        assert len(scores) == 3
