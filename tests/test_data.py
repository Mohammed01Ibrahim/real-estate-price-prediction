"""Target semantics and grouping are tested on invented, tiny datasets."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from real_estate.config import FEATURE_COLUMNS
from real_estate.data import prepare_data


def test_prepare_data_keeps_only_unambiguous_sale_usd_rows(synthetic_raw):
    frame = synthetic_raw.iloc[:15].copy()
    invalid_rows = {
        0: ("operation_type", "Alquiler"),
        1: ("currency", "PEN"),
        2: ("price_period", "Mensual"),
        3: ("l1", "Brasil"),
        4: ("property_type", "Oficina"),
        5: ("price", 0),
        6: ("price", -100),
        7: ("price", np.inf),
        8: ("surface_total", 9),
        9: ("surface_total", 10_001),
        10: ("surface_total", np.nan),
    }
    for index, (column, value) in invalid_rows.items():
        frame.loc[index, column] = value
    frame.loc[11, "price_period"] = "  "
    original = frame.copy(deep=True)
    features, target, groups, report = prepare_data(frame)
    assert len(features) == len(target) == len(groups) == 4
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features["surface_total"].tolist() == [105, 110, 115, 120]
    assert np.isfinite(target).all() and (target > 0).all()
    assert isinstance(report, dict)
    assert_frame_equal(frame, original)


def test_invalid_optional_training_values_become_missing(synthetic_raw):
    frame = synthetic_raw.iloc[:2].copy()
    frame["bathrooms"] = frame["bathrooms"].astype(object)
    frame.loc[0, "bedrooms"] = 3.5
    frame.loc[0, "bathrooms"] = True
    frame.loc[0, "lat"] = -90
    frame.loc[1, "lon"] = np.nan
    features, _, _, _ = prepare_data(frame)
    assert len(features) == 2
    assert features.iloc[0][["bedrooms", "bathrooms"]].isna().all()
    assert features[["lat", "lon"]].isna().all().all()
    assert "surface_covered" not in features and "rooms" not in features


def test_deduplication_ignores_listing_ids_and_groups_ignore_target(synthetic_raw):
    first = synthetic_raw.iloc[[0]].copy()
    duplicate = first.copy()
    duplicate["id"] = "synthetic-reposted"
    new_price = first.copy()
    new_price["id"] = "synthetic-repriced"
    new_price["price"] += 1_000
    features, target, groups, _ = prepare_data(pd.concat([first, duplicate, new_price], ignore_index=True))
    assert len(features) == 2
    assert target.nunique() == 2
    assert groups.nunique() == 1


def test_grouping_uses_rounded_coordinates(synthetic_raw):
    first = synthetic_raw.iloc[[0]].copy()
    nearby = first.copy()
    first["lat"], first["lon"] = -12.041, -77.031
    nearby["lat"], nearby["lon"] = -12.042, -77.032
    nearby["price"] += 100
    _, _, groups, _ = prepare_data(pd.concat([first, nearby], ignore_index=True))
    assert len(groups) == 2
    assert groups.nunique() == 1


def test_one_listing_id_cannot_cross_different_feature_groups(synthetic_raw):
    first = synthetic_raw.iloc[[0]].copy()
    changed = first.copy()
    changed["bedrooms"] += 1
    with pytest.raises(ValueError, match="Repeated listing IDs") as caught:
        prepare_data(pd.concat([first, changed], ignore_index=True))
    assert first.iloc[0]["id"] not in str(caught.value)


def test_same_listing_id_can_be_repriced_within_its_feature_group(synthetic_raw):
    first = synthetic_raw.iloc[[0]].copy()
    repriced = first.copy()
    repriced["price"] += 1_000
    _, prices, groups, _ = prepare_data(pd.concat([first, repriced], ignore_index=True))
    assert len(prices) == 2
    assert groups.nunique() == 1


def test_blank_listing_ids_are_not_treated_as_one_property(synthetic_raw):
    frame = synthetic_raw.iloc[:2].copy()
    frame["id"] = ""
    features, _, groups, _ = prepare_data(frame)
    assert len(features) == 2
    assert groups.nunique() == 2


@pytest.mark.parametrize("required_field", ["id", "operation_type", "currency", "price_period", "l1", "price", "surface_total", "property_type"])
def test_missing_raw_schema_is_a_clear_failure(synthetic_raw, required_field):
    with pytest.raises(ValueError, match=required_field):
        prepare_data(synthetic_raw.drop(columns=required_field))
