"""Flask contract and safe failure responses use tiny synthetic artifacts."""

import pytest

from real_estate.pipeline import build_pipeline
from real_estate.predict import save_artifact
from real_estate.web import create_app


@pytest.fixture
def app(model_path):
    app = create_app(model_path=model_path)
    app.config["TESTING"] = True
    return app


def test_get_form_static_assets_and_health_are_ready(app):
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Peru residential" in response.data
    assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/health").get_json() == {"status": "ready"}
    assert app.debug is False


def test_valid_json_uses_fitted_predictor_and_usd_contract(app, valid_payload):
    client = app.test_client()
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 200
    result = response.get_json()
    assert result["currency"] == "USD"
    assert result["target"] == "residential sale asking price"
    assert isinstance(result["prediction_usd"], float)
    assert result["prediction_usd"] > 0
    predictor = app.extensions["real_estate_predictor"]
    assert result["prediction_usd"] == predictor.predict(valid_payload)


def test_valid_minimal_form_predicts_with_optional_fields_blank(app):
    response = app.test_client().post("/", data={"surface_total": "100", "property_type": "Casa"})
    assert response.status_code == 200
    assert b"USD" in response.data
    assert b"Unable to estimate" not in response.data


@pytest.mark.parametrize("change", [{"surface_total": -1}, {"bedrooms": 2.2}, {"property_type": "Oficina"}, {"lat": None}, {"price": 100_000}])
def test_invalid_json_requests_return_user_error(app, valid_payload, change):
    response = app.test_client().post("/api/predict", json=dict(valid_payload, **change))
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_invalid_form_values_are_escaped(app):
    malicious_value = '\"><script>alert("synthetic")</script>'
    response = app.test_client().post("/", data={"surface_total": malicious_value, "property_type": "Casa"})
    assert response.status_code == 400
    assert b"<script>" not in response.data
    assert b"&lt;script&gt;" in response.data


@pytest.mark.parametrize("body,content_type", [("{broken-json", "application/json"), ("[]", "application/json"), ("null", "application/json"), ("surface_total=100", "text/plain")])
def test_malformed_or_non_object_json_is_rejected(app, body, content_type):
    response = app.test_client().post("/api/predict", data=body, content_type=content_type)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_oversized_requests_are_bounded(app):
    response = app.test_client().post("/api/predict", data="x" * (17 * 1024), content_type="application/json")
    assert response.status_code == 413
    assert "error" in response.get_json()


def test_missing_artifact_is_graceful_and_training_after_startup_recovers(tmp_path, fitted_dummy, valid_payload):
    model_path = tmp_path / "initially-missing.joblib"
    app = create_app(model_path)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 503
    unavailable = client.post("/api/predict", json=valid_payload)
    assert unavailable.status_code == 503
    assert str(model_path) not in unavailable.get_json()["error"]
    assert client.post("/", data=valid_payload).status_code == 503
    save_artifact(fitted_dummy, model_path, metadata={"test_fixture": "synthetic"})
    assert client.get("/health").status_code == 200
    assert client.post("/api/predict", json=valid_payload).status_code == 200


def test_unfitted_artifact_is_not_ready_and_does_not_serve_predictions(tmp_path, valid_payload):
    model_path = tmp_path / "unfitted.joblib"
    save_artifact(build_pipeline("dummy"), model_path, metadata={"test_fixture": "synthetic"})
    client = create_app(model_path).test_client()
    assert client.get("/health").status_code == 503
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 503
    assert "error" in response.get_json()


def test_unexpected_prediction_error_does_not_expose_exception(app, valid_payload):
    class BrokenPredictor:
        def predict(self, payload):
            raise RuntimeError("synthetic private exception detail")

    app.extensions["real_estate_predictor"] = BrokenPredictor()
    client = app.test_client()
    for response in [client.post("/api/predict", json=valid_payload), client.post("/", data=valid_payload)]:
        assert response.status_code == 500
        assert b"synthetic private exception detail" not in response.data
        assert b"could not be completed" in response.data


def test_app_templates_and_model_work_from_another_directory(model_path, valid_payload, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = create_app(model_path=model_path).test_client()
    assert client.get("/").status_code == 200
    assert client.post("/api/predict", json=valid_payload).status_code == 200
