"""Prepare and cache a local, allowlisted exploration artifact."""

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from threading import RLock

import numpy as np
import pandas as pd

from .config import DATA_PATH, EXPLORER_PATH, FEATURE_COLUMNS, NUMERIC_BOUNDS, PROPERTY_TYPES
from .data import prepare_data, read_raw
from .evaluate import regression_metrics
from .filters import MAX_EXPLORER_LISTINGS
from .predict import Predictor

LISTING_FIELDS = ("price_usd", "surface_total", "bedrooms", "bathrooms", "property_type", "l2", "l4", "lat", "lon")
UNAVAILABLE_EVALUATION = "Verified evaluation predictions are unavailable. Train the model and rebuild the explorer data."


class MissingExplorerData(RuntimeError):
    """The prepared local artifact is absent or incompatible."""


def safe_location(value) -> str | None:
    """Administrative labels only; discard contact-like or invalid content."""
    if pd.isna(value):
        return None
    label = str(value).strip()
    if (not label or len(label) > 100 or any(ord(c) < 32 for c in label)
            or "@" in label or re.search(r"(?:\d[\s().+-]*){8,}", label)):
        return None
    return label


def _json_rows(frame: pd.DataFrame) -> list[dict]:
    # pandas performs consistent null conversion, avoiding NaN/Infinity JSON.
    return json.loads(frame.to_json(orient="records", double_precision=10))


def _evaluation(model_path, evaluation_path, predictions_path, digest, X, y, groups) -> dict:
    if not all((model_path, evaluation_path, predictions_path)):
        return {"available": False, "reason": UNAVAILABLE_EVALUATION}
    try:
        from .train import make_splits
        predictor = Predictor(model_path)
        report = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
        if report["dataset_sha256"] != digest or predictor.metadata["dataset_sha256"] != digest:
            raise ValueError("Evaluation dataset mismatch.")
        if predictor.metadata["selected_model"] != report["selected_model"]:
            raise ValueError("Selected model mismatch.")
        # The saved estimator carries the run that produced its diagnostics.
        # Older trusted artifacts may omit optional report metadata; never fill
        # missing provenance with invented values, but verify every saved field.
        for name in ("seed", "run_utc", "comparison", "permutation_importance_validation", "held_out_test"):
            if name in predictor.metadata and predictor.metadata[name] != report.get(name):
                raise ValueError("Evaluation run metadata mismatch.")
        saved_split = predictor.metadata.get("split")
        if saved_split is not None:
            for name in ("rows", "groups", "method"):
                if name in saved_split and saved_split[name] != report["split"].get(name):
                    raise ValueError("Evaluation split metadata mismatch.")
        train_indices, validation_indices, test_indices = make_splits(X, y, groups, report["seed"])
        partitions = {"train": train_indices, "validation": validation_indices, "test": test_indices}
        split_rows = {name: len(indices) for name, indices in partitions.items()}
        if report["split"]["rows"] != split_rows:
            raise ValueError("Evaluation split row mismatch.")
        if "groups" in report["split"]:
            split_groups = {name: int(groups.iloc[indices].nunique()) for name, indices in partitions.items()}
            if report["split"]["groups"] != split_groups:
                raise ValueError("Evaluation split group mismatch.")
        points = pd.read_csv(predictions_path, usecols=["actual_usd", "predicted_usd"])
        actual = points["actual_usd"].to_numpy(dtype=float)
        predicted = points["predicted_usd"].to_numpy(dtype=float)
        if (len(points) != report["split"]["rows"]["test"] or len(points) != len(test_indices)
                or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
                or (actual <= 0).any() or (predicted <= 0).any()
                or not np.allclose(actual, y.iloc[test_indices].to_numpy(), rtol=1e-10, atol=1e-6)
                or not np.allclose(predicted, predictor.model.predict(X.iloc[test_indices]), rtol=1e-10, atol=1e-6)):
            raise ValueError("Held-out prediction mismatch.")
        measured = regression_metrics(actual, predicted)
        if any(not math.isclose(measured[k], report["held_out_test"][k], rel_tol=1e-8, abs_tol=1e-6) for k in measured):
            raise ValueError("Held-out metric mismatch.")
        result = {
            "available": True, "reason": None, "selected_model": report["selected_model"],
            "held_out_test": measured, "split": {"rows": split_rows}, "comparison": report["comparison"],
            "dataset_sha256": digest, "run_utc": report["run_utc"],
            "importance": [{"feature": name, **values} for name, values in
                           report["permutation_importance_validation"].items()],
            "importance_partition": "validation", "evaluation_partition": "held-out test",
            "predictions": {"actual_usd": actual.tolist(), "predicted_usd": predicted.tolist(),
                            "residual_usd": (actual - predicted).tolist()},
        }
        # Re-serialize strictly before writing the private prepared artifact.
        _validate_evaluation(result, digest)
        return result
    except Exception:
        # No partial/old evaluation is silently presented as a matching final run.
        return {"available": False, "reason": UNAVAILABLE_EVALUATION}


def prepare_explorer(data_path: Path = DATA_PATH, output_path: Path = EXPLORER_PATH,
                     model_path=None, evaluation_path=None, predictions_path=None) -> dict:
    """Reuse deterministic scope rules; never train or fit any transformation."""
    X, y, groups, report = prepare_data(read_raw(data_path))
    with Path(data_path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    frame = X.copy()
    frame["price_usd"] = y
    frame[["lat", "lon"]] = frame[["lat", "lon"]].round(2)
    for column in ("l2", "l4"):
        frame[column] = frame[column].map(safe_location)
    artifact = {
        "schema_version": 1,
        "metadata": {"dataset_sha256": digest, "coordinate_precision": 2, "currency": "USD",
                     "scope": "Peru; Venta; USD; blank price_period; Casa/Departamento; total area 10-10000 m2",
                     "generated_utc": datetime.now(timezone.utc).isoformat(), "retained_rows": len(frame)},
        "listings": _json_rows(frame.loc[:, LISTING_FIELDS]),
        "evaluation": _evaluation(model_path, evaluation_path, predictions_path, digest, X, y, groups),
    }
    validate_artifact(artifact)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    return artifact


def validate_artifact(artifact: dict) -> None:
    if (not isinstance(artifact, dict) or artifact.get("schema_version") != 1
            or set(artifact) != {"schema_version", "metadata", "listings", "evaluation"}
            or not isinstance(artifact.get("metadata"), dict)
            or artifact["metadata"].get("coordinate_precision") != 2
            or artifact["metadata"].get("currency") != "USD"
            or not re.fullmatch(r"[0-9a-f]{64}", str(artifact["metadata"].get("dataset_sha256", "")))
            or not isinstance(artifact.get("listings"), list)):
        raise ValueError("Invalid explorer artifact contract.")
    metadata = artifact["metadata"]
    if set(metadata) != {"dataset_sha256", "coordinate_precision", "currency", "scope", "generated_utc", "retained_rows"}:
        raise ValueError("Unsupported explorer metadata fields.")
    if metadata["scope"] != "Peru; Venta; USD; blank price_period; Casa/Departamento; total area 10-10000 m2":
        raise ValueError("Invalid explorer scope.")
    datetime.fromisoformat(metadata["generated_utc"])
    if isinstance(metadata["retained_rows"], bool) or not isinstance(metadata["retained_rows"], int):
        raise ValueError("Invalid explorer row count.")
    if len(artifact["listings"]) > MAX_EXPLORER_LISTINGS or artifact["metadata"].get("retained_rows") != len(artifact["listings"]):
        raise ValueError("Invalid explorer row count.")
    for row in artifact["listings"]:
        if not isinstance(row, dict) or set(row) != set(LISTING_FIELDS) or row["property_type"] not in PROPERTY_TYPES:
            raise ValueError("Invalid listing fields.")
        for name in ("price_usd", "surface_total", "lat", "lon", "bedrooms", "bathrooms"):
            value = row[name]
            if value is None and name not in {"price_usd", "surface_total"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Invalid listing numeric value.")
            if name == "price_usd":
                if value <= 0:
                    raise ValueError("Invalid price.")
            elif not NUMERIC_BOUNDS[name][0] <= value <= NUMERIC_BOUNDS[name][1]:
                raise ValueError("Invalid listing bounds.")
            if name in {"bedrooms", "bathrooms"} and int(value) != value:
                raise ValueError("Invalid listing count.")
            if name in {"lat", "lon"} and not math.isclose(round(value, 2), value, abs_tol=1e-10):
                raise ValueError("Coordinates must be rounded.")
        if (row["lat"] is None) != (row["lon"] is None):
            raise ValueError("Incomplete coordinate pair.")
        for name in ("l2", "l4"):
            if row[name] is not None and (not isinstance(row[name], str) or safe_location(row[name]) != row[name]):
                raise ValueError("Invalid administrative label.")
    evaluation = artifact.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("available"), bool):
        raise ValueError("Invalid evaluation contract.")
    _validate_evaluation(evaluation, metadata["dataset_sha256"])
    # A strict JSON roundtrip also rejects nonfinite values anywhere in metadata.
    json.dumps(artifact, allow_nan=False)


def _number(value, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError("Invalid evaluation numeric value.")


def _metrics(values):
    if not isinstance(values, dict) or set(values) != {"mae_usd", "rmse_usd", "r2"}:
        raise ValueError("Invalid evaluation metrics.")
    for name, value in values.items():
        _number(value, nonnegative=name != "r2")


def _validate_evaluation(evaluation, digest):
    from .pipeline import MODEL_NAMES
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("available"), bool):
        raise ValueError("Invalid evaluation contract.")
    if not evaluation["available"]:
        if set(evaluation) != {"available", "reason"} or evaluation["reason"] != UNAVAILABLE_EVALUATION:
            raise ValueError("Invalid unavailable evaluation.")
        return
    expected = {"available", "reason", "selected_model", "held_out_test", "split", "comparison", "dataset_sha256",
                "run_utc", "importance", "importance_partition", "evaluation_partition", "predictions"}
    if (set(evaluation) != expected or evaluation["reason"] is not None
            or evaluation["dataset_sha256"] != digest or evaluation["selected_model"] not in MODEL_NAMES
            or evaluation["importance_partition"] != "validation" or evaluation["evaluation_partition"] != "held-out test"):
        raise ValueError("Unsupported evaluation fields.")
    datetime.fromisoformat(evaluation["run_utc"])
    _metrics(evaluation["held_out_test"])
    split = evaluation["split"]
    if not isinstance(split, dict) or set(split) != {"rows"} or set(split["rows"]) != {"train", "validation", "test"}:
        raise ValueError("Invalid evaluation split.")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in split["rows"].values()):
        raise ValueError("Invalid evaluation split counts.")
    comparison = evaluation["comparison"]
    if not isinstance(comparison, list) or not 1 <= len(comparison) <= len(MODEL_NAMES):
        raise ValueError("Invalid model comparison.")
    names = set()
    for candidate in comparison:
        if (not isinstance(candidate, dict) or set(candidate) != {"model", "validation", "cross_validation"}
                or candidate["model"] not in MODEL_NAMES or candidate["model"] in names):
            raise ValueError("Invalid model comparison fields.")
        names.add(candidate["model"])
        _metrics(candidate["validation"])
        cv = candidate["cross_validation"]
        if not isinstance(cv, dict) or set(cv) != {"mae_usd", "rmse_usd", "r2"}:
            raise ValueError("Invalid cross-validation fields.")
        for name, values in cv.items():
            if not isinstance(values, dict) or set(values) != {"folds", "mean", "std"} or len(values["folds"]) != 3:
                raise ValueError("Invalid cross-validation values.")
            for value in values["folds"]:
                _number(value, name != "r2")
            _number(values["mean"], name != "r2")
            _number(values["std"], True)
    if evaluation["selected_model"] not in names:
        raise ValueError("Selected model absent from comparison.")
    importance = evaluation["importance"]
    if not isinstance(importance, list) or {row.get("feature") for row in importance} != set(FEATURE_COLUMNS) or len(importance) != len(FEATURE_COLUMNS):
        raise ValueError("Invalid importance feature contract.")
    for row in importance:
        if set(row) != {"feature", "mean_mae_increase_usd", "std_mae_increase_usd"}:
            raise ValueError("Invalid importance fields.")
        _number(row["mean_mae_increase_usd"])
        _number(row["std_mae_increase_usd"], True)
    points = evaluation["predictions"]
    if not isinstance(points, dict) or set(points) != {"actual_usd", "predicted_usd", "residual_usd"}:
        raise ValueError("Invalid held-out prediction fields.")
    for name, values in points.items():
        if not isinstance(values, list) or len(values) != split["rows"]["test"] or len(values) > 100000:
            raise ValueError("Invalid held-out prediction count.")
        for value in values:
            _number(value)
            if name != "residual_usd" and value <= 0:
                raise ValueError("Invalid held-out price.")
    if not np.allclose(np.asarray(points["actual_usd"]) - np.asarray(points["predicted_usd"]), points["residual_usd"], rtol=1e-10, atol=1e-6):
        raise ValueError("Invalid residual values.")
    measured = regression_metrics(points["actual_usd"], points["predicted_usd"])
    if any(not math.isclose(measured[k], evaluation["held_out_test"][k], rel_tol=1e-8, abs_tol=1e-6) for k in measured):
        raise ValueError("Invalid held-out metrics.")


class ExplorerStore:
    """Reload on artifact change and cache at most 32 filtered responses."""

    def __init__(self, path: Path | str = EXPLORER_PATH):
        self.path = Path(path)
        self._token = None
        self._artifact = None
        self._cache = OrderedDict()
        self._lock = RLock()

    def get(self) -> dict:
        with self._lock:
            try:
                stat = self.path.stat()
                if stat.st_size > 100 * 1024**2:
                    raise ValueError("Explorer artifact exceeds the allowed size.")
                token = (stat.st_mtime_ns, stat.st_size)
                if token != self._token:
                    artifact = json.loads(self.path.read_text(encoding="utf-8"))
                    validate_artifact(artifact)
                    self._artifact, self._token = artifact, token
                    self._cache.clear()
                return self._artifact
            except Exception as exc:
                raise MissingExplorerData("Exploration data is unavailable. Prepare authorized local data using the README instructions.") from exc

    def response(self, kind, filters):
        from .analytics import insights_response, listings_response
        with self._lock:
            artifact = self.get()
            key = (kind, filters)
            if key not in self._cache:
                function = {"listings": listings_response, "insights": insights_response}[kind]
                self._cache[key] = function(artifact, filters)
                if len(self._cache) > 32:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            return self._cache[key]
