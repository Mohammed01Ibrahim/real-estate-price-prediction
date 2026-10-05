"""A versioned artifact serves the same estimator from any working directory."""

import joblib
import numpy as np
import pytest

from real_estate.config import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, PROJECT_ROOT
from real_estate.pipeline import build_pipeline
from real_estate.predict import ModelUnavailable, Predictor, save_artifact
from real_estate.validation import InputValidationError, validate_payload


def test_artifact_roundtrip_preserves_training_inference_prediction(model_path, fitted_dummy, valid_payload):
    expected = fitted_dummy.predict(validate_payload(valid_payload))[0]
    predictor = Predictor(model_path)
    observed = predictor.predict(valid_payload)
    assert isinstance(observed, float)
    assert observed == pytest.approx(expected)
    assert observed > 0
    assert predictor.metadata["test_fixture"] == "synthetic"
    artifact = joblib.load(model_path)
    assert artifact["schema_version"] == 1
    assert artifact["feature_columns"] == list(FEATURE_COLUMNS)


def test_unknown_category_and_missing_optional_features_predict(model_path):
    payload = {"surface_total": 100, "property_type": "Casa", "l4": "Previously unseen district"}
    assert np.isfinite(Predictor(model_path).predict(payload))


def test_predictor_preserves_validation_error(model_path, valid_payload):
    with pytest.raises(InputValidationError):
        Predictor(model_path).predict(dict(valid_payload, bedrooms=-1))


def test_paths_and_loaded_artifact_are_independent_of_working_directory(model_path, valid_payload, monkeypatch, tmp_path):
    expected = Predictor(model_path).predict(valid_payload)
    other_directory = tmp_path / "other-working-directory"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    assert PROJECT_ROOT.is_absolute() and DATA_PATH.is_absolute() and MODEL_PATH.is_absolute()
    assert Predictor(model_path).predict(valid_payload) == expected


def test_missing_and_corrupt_artifacts_fail_with_safe_message(tmp_path):
    missing = tmp_path / "missing.joblib"
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_text("This is deliberately not a model artifact.", encoding="utf-8")
    for path in [missing, corrupt]:
        with pytest.raises(ModelUnavailable, match="model is unavailable") as caught:
            Predictor(path)
        assert str(path) not in str(caught.value)


def test_unfitted_schema_valid_artifact_is_unavailable(tmp_path):
    path = tmp_path / "unfitted.joblib"
    save_artifact(build_pipeline("dummy"), path, metadata={"test_fixture": "synthetic"})
    with pytest.raises(ModelUnavailable):
        Predictor(path)


def test_estimator_prediction_failure_is_a_safe_model_unavailable_error(model_path, valid_payload, monkeypatch):
    predictor = Predictor(model_path)

    def fail(features):
        raise RuntimeError("synthetic private estimator failure")

    monkeypatch.setattr(predictor.model, "predict", fail)
    with pytest.raises(ModelUnavailable) as caught:
        predictor.predict(valid_payload)
    assert "synthetic private estimator failure" not in str(caught.value)


@pytest.mark.parametrize("change", ["schema", "features", "sklearn_version", "model", "metadata"])
def test_incompatible_artifact_contract_is_rejected(model_path, change):
    artifact = joblib.load(model_path)
    if change == "schema":
        artifact["schema_version"] = 0
    elif change == "features":
        artifact["feature_columns"] = list(reversed(FEATURE_COLUMNS))
    elif change == "sklearn_version":
        artifact["metadata"]["sklearn_version"] = "0.0.0"
    elif change == "model":
        artifact["model"] = None
    else:
        artifact["metadata"] = None
    joblib.dump(artifact, model_path)
    with pytest.raises(ModelUnavailable):
        Predictor(model_path)


@pytest.mark.parametrize("bad_output", [[np.nan], [np.inf], [-10], [0], [1, 2]])
def test_invalid_model_output_never_becomes_a_price(model_path, valid_payload, monkeypatch, bad_output):
    predictor = Predictor(model_path)
    monkeypatch.setattr(predictor.model, "predict", lambda features: np.asarray(bad_output))
    with pytest.raises(ModelUnavailable):
        predictor.predict(valid_payload)
