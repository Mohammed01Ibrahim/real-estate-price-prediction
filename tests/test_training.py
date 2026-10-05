"""End-to-end, inexpensive training on generated fictional listings."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from real_estate.data import prepare_data, read_raw
from real_estate.predict import Predictor
from real_estate.train import make_splits, train


def test_group_splits_are_disjoint_reproducible_and_complete(synthetic_raw):
    features, prices, groups, _ = prepare_data(synthetic_raw)
    # Repeat each fictional property to exercise grouping rather than only row splitting.
    features = pd.concat([features, features], ignore_index=True)
    prices = pd.concat([prices, prices + 1_000], ignore_index=True)
    groups = pd.concat([groups, groups], ignore_index=True)
    splits = make_splits(features, prices, groups, seed=42)
    repeated = make_splits(features, prices, groups, seed=42)
    sets = [set(groups.iloc[indices]) for indices in splits]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert sorted(np.concatenate(splits)) == list(range(len(features)))
    assert [len(indices) for indices in splits] == [72, 24, 24]
    for actual, expected in zip(splits, repeated):
        np.testing.assert_array_equal(actual, expected)


def test_too_little_data_has_an_actionable_training_error(synthetic_raw):
    features, prices, groups, _ = prepare_data(synthetic_raw.iloc[:10])
    with pytest.raises(ValueError, match="at least 30 rows"):
        make_splits(features, prices, groups)


def test_raw_loader_excludes_unneeded_listing_text(synthetic_raw, tmp_path):
    raw = synthetic_raw.copy()
    raw["description"] = "Synthetic extra text not used by the model"
    raw["address"] = "Synthetic private field"
    path = tmp_path / "synthetic.csv"
    raw.to_csv(path, index=False)
    loaded = read_raw(path)
    assert "description" not in loaded and "address" not in loaded
    assert len(loaded) == len(raw)


def test_reduced_training_produces_usable_model_and_honest_usd_metrics(synthetic_raw, valid_payload, tmp_path, monkeypatch):
    data_path = tmp_path / "synthetic.csv"
    model_path = tmp_path / "models" / "synthetic.joblib"
    output_dir = tmp_path / "evaluation"
    synthetic_raw.to_csv(data_path, index=False)
    monkeypatch.chdir(tmp_path)
    report = train(data_path, model_path, output_dir, seed=42, model_names=("dummy", "ridge"))
    assert model_path.is_file()
    assert np.isfinite(Predictor(model_path).predict(valid_payload))
    assert report["dataset_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert [row["model"] for row in report["comparison"]] == ["dummy", "ridge"]
    best_validation = min(report["comparison"], key=lambda row: row["validation"]["mae_usd"])
    assert report["selected_model"] == best_validation["model"]
    assert report["split"]["rows"] == {"train": 36, "validation": 12, "test": 12}
    for row in report["comparison"]:
        assert len(row["cross_validation"]["mae_usd"]["folds"]) == 3
    persisted_report = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert persisted_report == report
    predictions = pd.read_csv(output_dir / "test_predictions.csv")
    actual, estimated = predictions["actual_usd"], predictions["predicted_usd"]
    assert report["held_out_test"]["mae_usd"] == pytest.approx(mean_absolute_error(actual, estimated))
    assert report["held_out_test"]["rmse_usd"] == pytest.approx(np.sqrt(mean_squared_error(actual, estimated)))
    for filename in ["prediction_vs_actual.png", "residual_analysis.png", "permutation_importance.png"]:
        assert (output_dir / filename).stat().st_size > 0
    features, prices, groups, _ = prepare_data(read_raw(data_path))
    training_indices, validation_indices, _ = make_splits(features, prices, groups)
    development_indices = np.concatenate([training_indices, validation_indices])
    model = Predictor(model_path).model
    pipeline = model.regressor_ if hasattr(model, "regressor_") else model
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == np.median(features.iloc[development_indices]["surface_total"])
