"""Feature mathematics, missing values, and fitted-preprocessor isolation."""

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from real_estate.config import ENGINEERED_NUMERIC_COLUMNS, FEATURE_COLUMNS
from real_estate.features import FeatureEngineer
from real_estate.pipeline import build_pipeline
from real_estate.validation import validate_payload


def _pipeline(estimator):
    return estimator.regressor_ if hasattr(estimator, "regressor_") else estimator


def test_engineering_preserves_shared_schema_without_unsupported_features(valid_payload):
    original = validate_payload(valid_payload)
    saved = original.copy(deep=True)
    transformed = FeatureEngineer().fit_transform(original)
    assert list(transformed.columns) == [*FEATURE_COLUMNS, "distance_to_lima_km"]
    assert not {"surface_covered", "rooms", "area_ratio", "area_per_room"} & set(transformed.columns)
    assert_frame_equal(original, saved)


def test_geographic_distance_uses_coarse_coordinates_and_haversine(valid_payload):
    features = validate_payload(dict(valid_payload, lat=-12.0464, lon=-77.0428))
    row = FeatureEngineer().fit_transform(features).iloc[0]
    assert row["lat"] == -12.05
    assert row["lon"] == -77.04
    latitude, longitude = map(math.radians, [-12.05, -77.04])
    anchor_latitude, anchor_longitude = map(math.radians, [-12.0464, -77.0428])
    a = math.sin((latitude - anchor_latitude) / 2) ** 2
    a += math.cos(anchor_latitude) * math.cos(latitude) * math.sin((longitude - anchor_longitude) / 2) ** 2
    expected = 2 * 6371.0088 * math.asin(math.sqrt(a))
    assert row["distance_to_lima_km"] == pytest.approx(expected)
    assert 0 < row["distance_to_lima_km"] < 1


def test_missing_optional_values_propagate_until_fitted_imputation():
    features = validate_payload({"surface_total": 100, "property_type": "Casa"})
    row = FeatureEngineer().fit_transform(features).iloc[0]
    assert pd.isna(row["distance_to_lima_km"])


@pytest.mark.parametrize("model_name", ["dummy", "ridge"])
def test_entirely_missing_optional_columns_still_allow_inference(synthetic_features, synthetic_prices, model_name):
    training = synthetic_features.copy()
    nullable = [field for field in FEATURE_COLUMNS if field not in {"surface_total", "property_type"}]
    training[nullable] = np.nan
    estimator = build_pipeline(model_name).fit(training, synthetic_prices)
    inference = validate_payload({"surface_total": 100, "property_type": "Casa", "l4": "Unseen district"})
    prediction = estimator.predict(inference)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    assert prediction[0] > 0


def test_preprocessors_learn_only_training_rows(synthetic_features, synthetic_prices):
    training = synthetic_features.iloc[:30].copy()
    heldout = synthetic_features.iloc[30:].copy()
    heldout["surface_total"] = 9_999
    heldout["l4"] = "Only in held-out data"
    estimator = build_pipeline("ridge").fit(training, synthetic_prices.iloc[:30])
    pipeline = _pipeline(estimator)
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    categorical = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    total_index = list(ENGINEERED_NUMERIC_COLUMNS).index("surface_total")
    assert numeric.named_steps["impute"].statistics_[total_index] == np.median(training["surface_total"])
    assert "Only in held-out data" not in categorical.named_steps["encode"].categories_[2]
    before = numeric.named_steps["impute"].statistics_.copy()
    prediction = estimator.predict(heldout)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(numeric.named_steps["impute"].statistics_, before)


def test_each_cross_validation_fold_fits_its_own_statistics(synthetic_features, synthetic_prices):
    groups = np.repeat(np.arange(20), 3)
    template = build_pipeline("ridge")
    for train_indices, validation_indices in GroupKFold(n_splits=3).split(synthetic_features, synthetic_prices, groups):
        estimator = clone(template).fit(synthetic_features.iloc[train_indices], synthetic_prices.iloc[train_indices])
        preprocessor = _pipeline(estimator).named_steps["preprocess"]
        numeric = preprocessor.named_transformers_["numeric"].named_steps["impute"]
        assert numeric.statistics_[0] == np.median(synthetic_features.iloc[train_indices]["surface_total"])
        assert not set(groups[train_indices]) & set(groups[validation_indices])
        assert np.isfinite(estimator.predict(synthetic_features.iloc[validation_indices])).all()
    assert not hasattr(template, "regressor_")


def test_baseline_is_median_on_original_usd_scale(synthetic_features):
    target = pd.Series([1.0, 10.0, 100.0, 1_000.0])
    estimator = build_pipeline("dummy").fit(synthetic_features.iloc[:4], target)
    assert estimator.predict(synthetic_features.iloc[[0]])[0] == 55


def test_ridge_target_transform_returns_original_price_scale(synthetic_features, synthetic_prices):
    estimator = build_pipeline("ridge").fit(synthetic_features, synthetic_prices)
    inference = synthetic_features.iloc[[10]]
    inner_prediction = estimator.regressor_.predict(inference)
    np.testing.assert_allclose(estimator.predict(inference), np.expm1(inner_prediction))


def test_unknown_model_name_is_a_clear_failure():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("undocumented-model")
