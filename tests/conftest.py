"""Synthetic test data; no source listings, addresses, or personal data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_estate.config import FEATURE_COLUMNS


@pytest.fixture
def valid_payload():
    return {
        "surface_total": 120,
        "bedrooms": 3,
        "bathrooms": 2,
        "lat": -12.05,
        "lon": -77.04,
        "property_type": "Departamento",
        "l2": "Lima",
        "l4": "Synthetic district",
    }


@pytest.fixture
def synthetic_features():
    """Generate independent fictional properties for fast pipeline checks."""
    rows = []
    for index in range(60):
        rows.append(
            {
                "surface_total": float(50 + index * 5),
                "lat": -12.0 - index * 0.01,
                "lon": -77.0 + index * 0.01,
                "bedrooms": float(1 + index % 4),
                "bathrooms": float(1 + index % 3),
                "property_type": "Casa" if index % 2 else "Departamento",
                "l2": "Synthetic region A" if index % 2 else "Synthetic region B",
                "l4": f"Synthetic district {index % 3}",
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


@pytest.fixture
def synthetic_prices(synthetic_features):
    return pd.Series(
        25_000 + synthetic_features["surface_total"].to_numpy() * 900,
        name="price",
    )


@pytest.fixture
def synthetic_raw(synthetic_features, synthetic_prices):
    frame = synthetic_features.copy(deep=True)
    frame.insert(0, "id", [f"synthetic-{index}" for index in range(len(frame))])
    frame["operation_type"] = "Venta"
    frame["currency"] = "USD"
    frame["price_period"] = pd.Series([None] * len(frame), dtype=object)
    # These legacy columns are retained in the input schema, not model inputs.
    frame["surface_covered"] = np.nan
    frame["rooms"] = np.nan
    frame["l1"] = "Perú"
    frame["price"] = synthetic_prices
    return frame


@pytest.fixture
def fitted_dummy(synthetic_features, synthetic_prices):
    from real_estate.pipeline import build_pipeline

    return build_pipeline("dummy", seed=42).fit(synthetic_features, synthetic_prices)


@pytest.fixture
def model_path(tmp_path, fitted_dummy):
    from real_estate.predict import save_artifact

    path = tmp_path / "synthetic-model.joblib"
    save_artifact(fitted_dummy, path, metadata={"test_fixture": "synthetic"})
    return path
