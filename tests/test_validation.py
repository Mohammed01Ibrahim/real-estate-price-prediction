"""Serving validation rejects malformed values before model inference."""

import numpy as np
import pandas as pd
import pytest

from real_estate.config import FEATURE_COLUMNS
from real_estate.validation import InputValidationError, validate_payload


def test_valid_payload_produces_training_feature_contract(valid_payload):
    frame = validate_payload(valid_payload)
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (1, len(FEATURE_COLUMNS))
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert frame.iloc[0]["surface_total"] == 120


def test_minimal_payload_marks_optional_fields_missing():
    frame = validate_payload({"surface_total": "85", "property_type": "Casa"})
    row = frame.iloc[0]
    assert row["surface_total"] == 85
    for field in set(FEATURE_COLUMNS) - {"surface_total", "property_type"}:
        assert pd.isna(row[field])


def test_form_numeric_strings_and_optional_blanks(valid_payload):
    payload = dict(valid_payload, surface_total="120.5", bathrooms="2", l4="  ")
    payload["bedrooms"] = None
    row = validate_payload(payload).iloc[0]
    assert row["surface_total"] == 120.5
    assert row["bathrooms"] == 2
    assert pd.isna(row["l4"])
    assert pd.isna(row["bedrooms"])


@pytest.mark.parametrize("value", [None, "", True, False, "bad", np.nan, np.inf, -np.inf, 0, 9, 10_001])
def test_invalid_required_area_is_rejected(valid_payload, value):
    with pytest.raises(InputValidationError):
        validate_payload(dict(valid_payload, surface_total=value))


@pytest.mark.parametrize("field,value", [("bedrooms", -1), ("bedrooms", 1.5), ("bathrooms", 51), ("bathrooms", True), ("bedrooms", np.inf)])
def test_invalid_optional_numbers_are_rejected(valid_payload, field, value):
    payload = dict(valid_payload)
    payload[field] = value
    with pytest.raises(InputValidationError):
        validate_payload(payload)


@pytest.mark.parametrize("field", ["bedrooms", "bathrooms", "lat"])
def test_numpy_booleans_are_not_numeric_values(valid_payload, field):
    with pytest.raises(InputValidationError):
        validate_payload(dict(valid_payload, **{field: np.bool_(True)}))


@pytest.mark.parametrize("value", [None, "", "Oficina", "casa", 3, True])
def test_unsupported_property_types_are_rejected(valid_payload, value):
    with pytest.raises(InputValidationError):
        validate_payload(dict(valid_payload, property_type=value))


@pytest.mark.parametrize("updates", [{"lat": -20}, {"lat": 2}, {"lon": -83}, {"lon": -67}, {"lat": None}, {"lon": ""}, {"lat": True}, {"lon": np.nan}])
def test_invalid_or_partial_coordinates_are_rejected(valid_payload, updates):
    with pytest.raises(InputValidationError):
        validate_payload(dict(valid_payload, **updates))


@pytest.mark.parametrize("latitude,longitude", [(-19, -82), (1, -68)])
def test_coordinate_bounds_are_inclusive(valid_payload, latitude, longitude):
    row = validate_payload(dict(valid_payload, lat=latitude, lon=longitude)).iloc[0]
    assert row["lat"] == latitude
    assert row["lon"] == longitude


@pytest.mark.parametrize("legacy_field", ["surface_covered", "rooms"])
def test_unavailable_legacy_features_are_not_public_inputs(valid_payload, legacy_field):
    assert legacy_field not in FEATURE_COLUMNS
    with pytest.raises(InputValidationError):
        validate_payload(dict(valid_payload, **{legacy_field: 100}))


@pytest.mark.parametrize("field,value", [("l2", 17), ("l4", ["district"]), ("l4", "x" * 101)])
def test_categories_require_bounded_text(valid_payload, field, value):
    with pytest.raises(InputValidationError):
        validate_payload(dict(valid_payload, **{field: value}))


def test_unknown_fields_do_not_silently_change_contract(valid_payload):
    with pytest.raises(InputValidationError):
        validate_payload(dict(valid_payload, price=500_000))


@pytest.mark.parametrize("payload", [None, [], "Casa", 120])
def test_payload_must_be_a_mapping(payload):
    with pytest.raises(InputValidationError):
        validate_payload(payload)
