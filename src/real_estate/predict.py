"""Versioned artifacts and validated predictions from the fitted estimator."""

from collections.abc import Mapping
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.utils.validation import check_is_fitted

from .config import FEATURE_COLUMNS, MODEL_PATH
from .validation import validate_payload


class ModelUnavailable(RuntimeError):
    """The local, trusted pipeline is missing, corrupt, or incompatible."""


def save_artifact(model, path: Path | str, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "schema_version": 1, "feature_columns": list(FEATURE_COLUMNS), "model": model,
        "metadata": {**metadata, "sklearn_version": sklearn.__version__, "python_version": platform.python_version()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(content, temporary, compress=3)
    temporary.replace(path)


class Predictor:
    """Load only trusted local artifacts; pickle/joblib is executable content."""

    def __init__(self, model_path: Path | str | None = None):
        self.path = Path(model_path) if model_path is not None else MODEL_PATH
        try:
            artifact = joblib.load(self.path)
            if (not isinstance(artifact, dict) or artifact.get("schema_version") != 1
                    or artifact.get("feature_columns") != list(FEATURE_COLUMNS)
                    or artifact.get("metadata", {}).get("sklearn_version") != sklearn.__version__
                    or not hasattr(artifact.get("model"), "predict")):
                raise ValueError("Incompatible artifact contract.")
            check_is_fitted(artifact["model"])
            self.model = artifact["model"]
            self.metadata = artifact["metadata"]
        except Exception as exc:
            raise ModelUnavailable("The model is unavailable. Train it with the documented command.") from exc

    def predict(self, payload: Mapping) -> float:
        features = validate_payload(payload)
        try:
            prediction = np.asarray(self.model.predict(features), dtype=float)
        except Exception as exc:
            raise ModelUnavailable("The model could not produce a valid price estimate.") from exc
        if prediction.shape != (1,) or not np.isfinite(prediction[0]) or prediction[0] <= 0:
            raise ModelUnavailable("The model could not produce a valid price estimate.")
        return float(prediction[0])
