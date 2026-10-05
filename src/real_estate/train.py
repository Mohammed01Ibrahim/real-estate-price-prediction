"""Train one canonical pipeline; choose by validation MAE, then test once."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from threadpoolctl import threadpool_limits

from .config import ARTIFACTS_PATH, DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, SEED
from .data import prepare_data, read_raw
from .evaluate import regression_metrics, write_figures
from .pipeline import MODEL_NAMES, build_pipeline
from .predict import save_artifact

LOGGER = logging.getLogger(__name__)


def make_splits(X, y, groups, seed: int = SEED):
    """60/20/20 approximate rows; group proportions are exact split controls."""
    if len(X) < 30 or groups.nunique() < 10:
        raise ValueError("Training requires at least 30 rows and 10 distinct listing groups.")
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
    train_rel, val_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1).split(
        X.iloc[development], y.iloc[development], groups.iloc[development]))
    return development[train_rel], development[val_rel], test


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH,
          output_dir: Path = ARTIFACTS_PATH, seed: int = SEED,
          model_names: tuple[str, ...] = MODEL_NAMES) -> dict:
    raw = read_raw(data_path)
    X, y, groups, data_report = prepare_data(raw)
    train_idx, val_idx, test_idx = make_splits(X, y, groups, seed)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    cv = GroupKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    comparisons, fitted = [], {}
    with threadpool_limits(limits=2):
        for name in model_names:
            LOGGER.info("Evaluating %s: three training-only CV folds and fixed validation partition", name)
            estimator = build_pipeline(name, seed)
            cv_result = cross_validate(estimator, X_train, y_train, groups=groups.iloc[train_idx],
                                       cv=cv, scoring=scores, n_jobs=1, error_score="raise")
            estimator.fit(X_train, y_train)
            validation = regression_metrics(y_val, estimator.predict(X_val))
            result = {"model": name, "validation": validation, "cross_validation": {}}
            for metric in scores:
                values = cv_result[f"test_{metric}"] * (-1 if metric != "r2" else 1)
                key = metric + ("_usd" if metric != "r2" else "")
                result["cross_validation"][key] = {"folds": values.tolist(),
                                                      "mean": float(values.mean()), "std": float(values.std())}
            comparisons.append(result)
            fitted[name] = estimator
            LOGGER.info("%s validation MAE $%.2f", name, validation["mae_usd"])
        selected = min(comparisons, key=lambda row: row["validation"]["mae_usd"])["model"]
        # Importance uses validation and the training-only fitted winner; never tunes on test.
        permutation = permutation_importance(fitted[selected], X_val, y_val,
                                             scoring="neg_mean_absolute_error", n_repeats=3,
                                             random_state=seed, n_jobs=1)
        importance = {
            name: {"mean_mae_increase_usd": float(mean), "std_mae_increase_usd": float(std)}
            for name, mean, std in zip(FEATURE_COLUMNS, permutation.importances_mean, permutation.importances_std)
        }
        final = clone(fitted[selected])
        development_idx = np.concatenate([train_idx, val_idx])
        final.fit(X.iloc[development_idx], y.iloc[development_idx])
        predicted = final.predict(X.iloc[test_idx])
        test_metrics = regression_metrics(y.iloc[test_idx], predicted)
    with Path(data_path).open("rb") as stream:
        dataset_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "dataset_sha256": dataset_hash, "data": data_report,
        "scope": "Peru; Venta; USD; blank price_period; Casa/Departamento; total area 10-10000 m2",
        "target": "Historical residential sale asking price in USD; blank period interpreted as total price",
        "split": {"method": "group-aware random 60/20/20; proxy groups from raw features with 0.01-degree coordinates",
                  "rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
                  "groups": {"train": int(groups.iloc[train_idx].nunique()),
                             "validation": int(groups.iloc[val_idx].nunique()), "test": int(groups.iloc[test_idx].nunique())}},
        "selection": "Lowest original-USD validation MAE; fixed candidates, no hyperparameter search",
        "comparison": comparisons, "selected_model": selected, "held_out_test": test_metrics,
        "permutation_importance_validation": importance,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "Flask", "joblib", "matplotlib")},
    }
    save_artifact(final, model_path, report)
    report["artifact_bytes"] = Path(model_path).stat().st_size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_figures(y.iloc[test_idx], predicted, output_dir, importance)
    # Never publish per-listing predictions; this diagnostic stays in ignored artifacts/.
    pd.DataFrame({"actual_usd": y.iloc[test_idx].to_numpy(), "predicted_usd": predicted}).to_csv(
        output_dir / "test_predictions.csv", index=False)
    LOGGER.info("Selected %s; test MAE $%.2f, RMSE $%.2f, R2 %.4f; artifact %.2f MiB",
                selected, test_metrics["mae_usd"], test_metrics["rmse_usd"], test_metrics["r2"],
                report["artifact_bytes"] / 1024**2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=ARTIFACTS_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        train(args.data, args.model, args.output, args.seed, tuple(args.models))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
