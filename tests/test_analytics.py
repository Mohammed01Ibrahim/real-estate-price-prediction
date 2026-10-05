"""Analytics preserve full filtered populations and disclose bounded samples."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from real_estate.analytics import insights_response, listings_response
from real_estate.explorer import prepare_explorer
from real_estate.filters import parse_filters
from real_estate.web import create_app


@pytest.fixture
def analytics_artifact(tmp_path, synthetic_raw):
    source, output = tmp_path / "fictional.csv", tmp_path / "analytics.json"
    synthetic_raw.to_csv(source, index=False)
    return prepare_explorer(source, output)


def test_summary_and_regions_use_current_scope_and_medians(analytics_artifact):
    result = insights_response(analytics_artifact, parse_filters({}))
    frame = pd.DataFrame(analytics_artifact["listings"])
    assert result["summary"] == {
        "listings": len(frame), "regions": frame["l2"].nunique(),
        "median_price_usd": float(frame["price_usd"].median()),
        "median_area_m2": float(frame["surface_total"].median()),
        "casa": 30, "departamento": 30,
    }
    assert {row["property_type"]: row["count"] for row in result["property_types"]} == {"Casa": 30, "Departamento": 30}
    regions = {row["region"]: row for row in result["regions"]}
    for name, group in frame.groupby("l2"):
        assert regions[name]["count"] == len(group)
        assert regions[name]["median_price_usd"] == float(group["price_usd"].median())


def test_price_histograms_include_extremes_and_all_records(analytics_artifact):
    artifact = deepcopy(analytics_artifact)
    artifact["listings"][-1]["price_usd"] = 1_000_000_000.0
    result = insights_response(artifact, parse_filters({}))
    prices = np.array([row["price_usd"] for row in artifact["listings"]])
    histogram = result["price_distribution"]
    assert sum(histogram["counts"]) == len(prices)
    assert sum(histogram["log_counts"]) == len(prices)
    for edges, counts in ((histogram["edges"], histogram["counts"]), (histogram["log_edges"], histogram["log_counts"])):
        assert len(edges) == len(counts) + 1
        assert edges[0] <= prices.min() and edges[-1] >= prices.max()
        assert np.all(np.diff(edges) > 0)
        assert counts[-1] > 0
    assert result["summary"]["median_price_usd"] == float(np.median(prices))
    assert max(row["price_usd"] for row in result["scatter"]["points"]) == prices.max()


def test_area_histogram_uses_m2_and_all_valid_scope_records(analytics_artifact):
    result = insights_response(analytics_artifact, parse_filters({}))
    histogram = result["area_distribution"]
    areas = [row["surface_total"] for row in analytics_artifact["listings"]]
    assert sum(histogram["counts"]) == len(areas)
    assert len(histogram["edges"]) == len(histogram["counts"]) + 1
    assert histogram["edges"][0] <= min(areas)
    assert histogram["edges"][-1] >= max(areas)


def test_analytics_share_map_filter_semantics_and_keep_global_evaluation(analytics_artifact):
    filtered = parse_filters({"property_type": "Casa", "min_area": "100", "max_area": "200"})
    result = insights_response(analytics_artifact, filtered)
    mapped = listings_response(analytics_artifact, filtered)
    expected = [row for row in analytics_artifact["listings"] if row["property_type"] == "Casa" and 100 <= row["surface_total"] <= 200]
    assert result["summary"]["listings"] == mapped["matched_listings"] == len(expected)
    assert result["summary"]["casa"] == len(expected)
    assert result["summary"]["departamento"] == 0
    assert result["summary"]["median_price_usd"] == float(np.median([row["price_usd"] for row in expected]))
    assert result["evaluation"] == analytics_artifact["evaluation"]
    assert result["evaluation"] == insights_response(analytics_artifact, parse_filters({}))["evaluation"]


def test_empty_filters_return_valid_json_and_undefined_aggregates(analytics_artifact):
    result = insights_response(analytics_artifact, parse_filters({"l2": "Unlisted region"}))
    assert result["summary"]["listings"] == result["summary"]["regions"] == 0
    assert result["summary"]["median_price_usd"] is None
    assert result["summary"]["median_area_m2"] is None
    assert sum(result["price_distribution"]["counts"]) == 0
    assert sum(result["area_distribution"]["counts"]) == 0
    assert result["scatter"]["points"] == []
    assert result["scatter"]["total"] == result["scatter"]["shown"] == 0
    assert result["scatter"]["sampled"] is False
    assert all(value is None for row in result["correlation"]["matrix"] for value in row)
    assert all(value == 0 for row in result["correlation"]["pair_counts"] for value in row)
    json.dumps(result, allow_nan=False)


def test_correlation_is_pairwise_and_uses_only_supported_numeric_variables(analytics_artifact):
    artifact = deepcopy(analytics_artifact)
    artifact["listings"][0]["bedrooms"] = None
    artifact["listings"][1]["lat"] = None
    artifact["listings"][1]["lon"] = None
    for row in artifact["listings"]:
        row["bathrooms"] = 2
    result = insights_response(artifact, parse_filters({}))["correlation"]
    expected_variables = ["surface_total", "bedrooms", "bathrooms", "lat", "lon", "price_usd"]
    assert result["variables"] == expected_variables
    frame = pd.DataFrame(artifact["listings"])[expected_variables]
    expected = frame.corr(method="pearson")
    for i, first in enumerate(expected_variables):
        for j, second in enumerate(expected_variables):
            count = int(frame[[first, second]].notna().all(axis=1).sum())
            assert result["pair_counts"][i][j] == count
            value = expected.iloc[i, j]
            if pd.isna(value):
                assert result["matrix"][i][j] is None
            else:
                assert result["matrix"][i][j] == pytest.approx(float(value))
    assert result["pair_counts"][1][3] == 58
    json.dumps(result, allow_nan=False)


def test_scatter_is_bounded_deterministic_and_retains_price_and_area_extremes(analytics_artifact):
    artifact = deepcopy(analytics_artifact)
    prototype = artifact["listings"][0]
    artifact["listings"] = [
        {**prototype, "price_usd": float(10_000 + index * 101), "surface_total": float(10 + index * 3)}
        for index in range(2500)
    ]
    # The extrema occur at separate positions so a naive prefix cannot retain them.
    artifact["listings"][2345]["price_usd"] = 2_000_000_000.0
    artifact["listings"][2300]["surface_total"] = 9999.0
    first = insights_response(artifact, parse_filters({}))
    second = insights_response(artifact, parse_filters({}))
    scatter = first["scatter"]
    assert scatter == second["scatter"]
    assert scatter["total"] == first["summary"]["listings"] == 2500
    assert scatter["sampled"] is True
    assert scatter["shown"] == len(scatter["points"]) <= 2000
    assert min(point["price_usd"] for point in scatter["points"]) == 10_000
    assert max(point["price_usd"] for point in scatter["points"]) == 2_000_000_000
    assert min(point["surface_total"] for point in scatter["points"]) == 10
    assert max(point["surface_total"] for point in scatter["points"]) == 9999
    assert sum(first["price_distribution"]["counts"]) == 2500
    assert sum(first["area_distribution"]["counts"]) == 2500
    assert all(set(point) == {"price_usd", "surface_total", "property_type", "l2", "l4"} for point in scatter["points"])


def test_pagination_does_not_change_analytic_population(analytics_artifact):
    first = insights_response(analytics_artifact, parse_filters({"limit": "1", "offset": "20"}))
    second = insights_response(analytics_artifact, parse_filters({}))
    assert first == second


def test_missing_optional_values_remain_unknown_without_imputation(analytics_artifact):
    artifact = deepcopy(analytics_artifact)
    for row in artifact["listings"]:
        row["bedrooms"] = row["bathrooms"] = None
        row["lat"] = row["lon"] = None
        row["l2"] = row["l4"] = None
    result = insights_response(artifact, parse_filters({}))
    assert result["summary"]["listings"] == 60
    assert result["summary"]["regions"] == 0
    assert all(point["l2"] is None and point["l4"] is None for point in result["scatter"]["points"])
    for column in (1, 2, 3, 4):
        assert all(row[column] is None for row in result["correlation"]["matrix"])
        assert all(row[column] == 0 for row in result["correlation"]["pair_counts"])
    json.dumps(result, allow_nan=False)


def test_api_valid_filters_return_finite_recomputed_insights(tmp_path, synthetic_raw):
    source, output = tmp_path / "fictional.csv", tmp_path / "analytics.json"
    synthetic_raw.to_csv(source, index=False)
    artifact = prepare_explorer(source, output)
    client = create_app(explorer_path=output).test_client()
    query = {"property_type": "Casa", "bedrooms": "2", "q": "region A"}
    response = client.get("/api/insights", query_string=query)
    assert response.status_code == 200
    expected = insights_response(artifact, parse_filters(query))
    assert response.get_json() == expected
    assert response.get_json()["summary"]["listings"] > 0
    json.dumps(response.get_json(), allow_nan=False)
