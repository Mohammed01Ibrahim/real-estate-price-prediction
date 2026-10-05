"""A small Flask interface that uses the same fitted pipeline as the CLI."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, Response, jsonify, render_template, request
from flask.typing import ResponseReturnValue
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

from .config import FEATURE_COLUMNS, MODEL_PATH, NUMERIC_BOUNDS, PROPERTY_TYPES
from .predict import ModelUnavailable, Predictor
from .validation import InputValidationError


PREDICTION_TARGET = "residential sale asking price"
MODEL_UNAVAILABLE_MESSAGE = (
    "The prediction model is unavailable. Train the model using the README "
    "instructions, then try again."
)
UNEXPECTED_ERROR_MESSAGE = "The prediction could not be completed. Please try again."


def create_app(model_path: str | Path | None = None) -> Flask:
    """Create an app that remains usable before its local model is trained.

    Predictor owns validation and preprocessing. No fitted transformations or
    categorical mappings are recreated in this application.
    """
    app = Flask(__name__)
    app.config.update(
        MODEL_PATH=Path(model_path) if model_path is not None else MODEL_PATH,
        MAX_CONTENT_LENGTH=16 * 1024,
    )
    app.json.sort_keys = False
    app.extensions["real_estate_predictor"] = None
    predictor_lock = Lock()

    def get_predictor() -> Predictor:
        # Failed loads are not cached: a model trained after startup can be used
        # on the next request without restarting the app.
        with predictor_lock:
            predictor = app.extensions["real_estate_predictor"]
            if predictor is None:
                predictor = Predictor(app.config["MODEL_PATH"])
                app.extensions["real_estate_predictor"] = predictor
            return predictor

    def page(
        *,
        values: dict[str, str] | None = None,
        prediction: float | None = None,
        error: str | None = None,
        status: int = 200,
    ) -> ResponseReturnValue:
        return (
            render_template(
                "index.html",
                values=values or {},
                property_types=PROPERTY_TYPES,
                numeric_bounds=NUMERIC_BOUNDS,
                prediction=(f"{prediction:,.0f}" if prediction is not None else None),
                error=error,
            ),
            status,
        )

    @app.before_request
    def enforce_body_limit() -> None:
        if (request.content_length is not None
                and request.content_length > app.config["MAX_CONTENT_LENGTH"]):
            raise RequestEntityTooLarge()

    @app.get("/")
    def index() -> ResponseReturnValue:
        return page()

    @app.post("/")
    def predict_form() -> ResponseReturnValue:
        values = {name: request.form.get(name, "") for name in FEATURE_COLUMNS}
        try:
            prediction = get_predictor().predict(values)
        except InputValidationError as exc:
            return page(values=values, error=str(exc), status=400)
        except ModelUnavailable:
            app.logger.warning("Prediction requested while the model is unavailable.")
            return page(values=values, error=MODEL_UNAVAILABLE_MESSAGE, status=503)
        except Exception:
            app.logger.exception("Unexpected failure during form prediction.")
            return page(values=values, error=UNEXPECTED_ERROR_MESSAGE, status=500)
        return page(values=values, prediction=prediction)

    @app.post("/api/predict")
    def predict_api() -> ResponseReturnValue:
        if not request.is_json:
            return jsonify(error="Send a JSON object with Content-Type application/json."), 400
        try:
            payload: Any = request.get_json()
        except (BadRequest, UnsupportedMediaType):
            return jsonify(error="The request body must contain valid JSON."), 400
        if not isinstance(payload, dict):
            return jsonify(error="The request body must be a JSON object."), 400
        try:
            prediction = get_predictor().predict(payload)
        except InputValidationError as exc:
            return jsonify(error=str(exc)), 400
        except ModelUnavailable:
            app.logger.warning("API prediction requested while the model is unavailable.")
            return jsonify(error=MODEL_UNAVAILABLE_MESSAGE), 503
        except Exception:
            app.logger.exception("Unexpected failure during API prediction.")
            return jsonify(error=UNEXPECTED_ERROR_MESSAGE), 500
        return jsonify(
            prediction_usd=prediction,
            currency="USD",
            target=PREDICTION_TARGET,
        )

    @app.get("/health")
    def health() -> ResponseReturnValue:
        try:
            get_predictor()
        except ModelUnavailable:
            return jsonify(status="not-ready"), 503
        except Exception:
            app.logger.exception("Unexpected failure while checking model readiness.")
            return jsonify(status="not-ready"), 503
        return jsonify(status="ready")

    @app.errorhandler(RequestEntityTooLarge)
    def body_too_large(_error: RequestEntityTooLarge) -> ResponseReturnValue:
        message = "The request is too large. The maximum body size is 16 KB."
        if request.path.startswith("/api/"):
            return jsonify(error=message), 413
        return page(error=message, status=413)

    @app.errorhandler(BadRequest)
    def malformed_request(_error: BadRequest) -> ResponseReturnValue:
        message = "The request could not be read. Please check its format."
        if request.path.startswith("/api/"):
            return jsonify(error=message), 400
        return page(error=message, status=400)

    @app.after_request
    def response_headers(response: Response) -> Response:
        # Submitted property details should not be stored in browser caches.
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app


def main() -> None:
    """Run the local portfolio demo without Flask's debugger or reloader."""
    create_app().run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
