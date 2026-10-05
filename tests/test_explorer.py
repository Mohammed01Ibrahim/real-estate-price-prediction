"""Prepared listing contracts use fictional records and never source data."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from real_estate.data import prepare_data
from real_estate.evaluate import regression_metrics
from real_estate.explorer import ExplorerStore, MissingExplorerData, prepare_explorer
from real_estate.filters import MAX_EXPLORER_LISTINGS, FilterValidationError, filter_listings, parse_filters
from real_estate.predict import save_artifact
from real_estate.train import make_splits
from real_estate.web import create_app


LISTING_FIELDS = {
    "price_usd", "surface_total", "bedrooms", "bathrooms",
    "property_type", "l2", "l4", "lat", "lon",
}


@pytest.fixture
def prepared_explorer(tmp_path, synthetic_raw):
    source = tmp_path / "fictional.csv"
    synthetic_raw.to_csv(source, index=False)
    output = tmp_path / "explorer.json"
    artifact = prepare_explorer(source, output)
    return artifact, output, source


@pytest.fixture
def synthetic_evaluation(tmp_path, synthetic_raw, fitted_dummy):
    """A matching tiny saved run exercises provenance without expensive training."""
    source = tmp_path / "evaluation-source.csv"
    synthetic_raw.to_csv(source, index=False)
    X, y, groups, _ = prepare_data(synthetic_raw)
    train_indices, val_indices, test_indices = make_splits(X, y, groups, 42)
    actual = y.iloc[test_indices].to_numpy()
    predicted = fitted_dummy.predict(X.iloc[test_indices])
    metrics = regression_metrics(actual, predicted)
    report = {
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_model": "dummy", "seed": 42,
        "run_utc": "2026-10-05T00:00:00+00:00",
        "split": {
            "method": "group-aware random 60/20/20",
            "rows": {"train": len(train_indices), "validation": len(val_indices), "test": len(test_indices)},
            "groups": {"train": int(groups.iloc[train_indices].nunique()), "validation": int(groups.iloc[val_indices].nunique()), "test": int(groups.iloc[test_indices].nunique())},
        },
        "held_out_test": metrics,
        "comparison": [{"model": "dummy", "validation": deepcopy(metrics), "cross_validation": {
            metric: {"folds": [value] * 3, "mean": value, "std": 0.0}
            for metric, value in metrics.items()
        }}],
        "permutation_importance_validation": {
            name: {"mean_mae_increase_usd": 0.0, "std_mae_increase_usd": 0.0}
            for name in X.columns
        },
    }
    model = tmp_path / "evaluation-model.joblib"
    evaluation = tmp_path / "evaluation.json"
    predictions = tmp_path / "test_predictions.csv"
    save_artifact(fitted_dummy, model, report)
    evaluation.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    pd.DataFrame({"actual_usd": actual, "predicted_usd": predicted}).to_csv(predictions, index=False)
    return {"source": source, "model": model, "evaluation": evaluation, "predictions": predictions,
            "report": report, "fitted_dummy": fitted_dummy, "actual": actual, "predicted": predicted}


def test_preparation_reuses_corrected_scope_without_fitting(tmp_path, synthetic_raw, monkeypatch):
    from sklearn.pipeline import Pipeline

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Explorer preparation must not fit a model or preprocessing.")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    raw = synthetic_raw.copy()
    raw.loc[0, "currency"] = "PEN"
    raw.loc[1, "operation_type"] = "Alquiler"
    raw.loc[2, "property_type"] = "Oficina"
    raw.loc[3, "price_period"] = "Mensual"
    raw.loc[4, "surface_total"] = 0
    raw.loc[5, "price"] = np.inf
    raw.loc[6, "l1"] = "Chile"
    source = tmp_path / "scope.csv"
    raw.to_csv(source, index=False)
    artifact = prepare_explorer(source, tmp_path / "explorer.json")
    expected_X, expected_y, _, _ = prepare_data(raw)
    assert len(artifact["listings"]) == len(expected_X)
    assert [row["price_usd"] for row in artifact["listings"]] == expected_y.tolist()
    assert {row["property_type"] for row in artifact["listings"]} == {"Casa", "Departamento"}
    assert artifact["evaluation"]["available"] is False
    assert artifact["evaluation"]["reason"]


def test_artifact_contains_only_safe_finite_fields(prepared_explorer):
    artifact, output, source = prepared_explorer
    assert artifact["schema_version"] == 1
    assert artifact["metadata"]["currency"] == "USD"
    assert artifact["metadata"]["coordinate_precision"] == 2
    assert artifact["metadata"]["dataset_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert artifact["metadata"]["scope"]
    assert all(set(row) == LISTING_FIELDS for row in artifact["listings"])
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    json.dumps(artifact, allow_nan=False)
    text = output.read_text(encoding="utf-8")
    assert "synthetic-" not in text
    assert str(source) not in text


def test_source_contact_columns_are_never_exported(tmp_path, synthetic_raw):
    raw = synthetic_raw.copy()
    private = "contact@example.test +51 999 888 777"
    for column in ("title", "description", "seller", "name", "email", "phone", "account_id"):
        raw[column] = private
    raw.loc[0, "l2"] = "contact@example.test"
    raw.loc[1, "l4"] = "+51 999 888 777"
    raw.loc[2, "l4"] = "Synthetic\x01 district"
    source = tmp_path / "privacy.csv"
    raw.to_csv(source, index=False)
    artifact = prepare_explorer(source, tmp_path / "safe.json")
    serialized = json.dumps(artifact, allow_nan=False)
    assert "contact@example.test" not in serialized
    assert "+51 999 888 777" not in serialized
    assert "\\u0001" not in serialized
    assert artifact["listings"][0]["l2"] is None
    assert artifact["listings"][1]["l4"] is None
    assert artifact["listings"][2]["l4"] is None
    assert all(set(row) == LISTING_FIELDS for row in artifact["listings"])


def test_coordinates_are_rounded_and_incomplete_pairs_are_retained_unmapped(tmp_path, synthetic_raw):
    raw = synthetic_raw.copy()
    raw.loc[0, ["lat", "lon"]] = [-12.05678, -77.04321]
    raw.loc[1, "lat"] = np.nan
    raw.loc[2, "lon"] = -100
    raw.loc[3, "lat"] = np.inf
    source = tmp_path / "coordinates.csv"
    raw.to_csv(source, index=False)
    artifact = prepare_explorer(source, tmp_path / "safe.json")
    rows = artifact["listings"]
    assert len(rows) == len(raw)
    assert rows[0]["lat"] == -12.06
    assert rows[0]["lon"] == -77.04
    for row in rows[1:4]:
        assert row["lat"] is None and row["lon"] is None
    assert "-12.05678" not in json.dumps(artifact)
    assert "-77.04321" not in json.dumps(artifact)


def test_store_reuses_cache_and_reloads_changed_artifact(prepared_explorer):
    artifact, output, _ = prepared_explorer
    store = ExplorerStore(output)
    initial = store.get()
    assert initial == artifact
    assert store.get() is initial
    changed = deepcopy(artifact)
    changed["listings"] = changed["listings"][:-1]
    changed["metadata"]["retained_rows"] = len(changed["listings"])
    output.write_text(json.dumps(changed, allow_nan=False), encoding="utf-8")
    assert store.get() == changed
    assert store.get() is not initial


@pytest.mark.parametrize("corruption", ["invalid_json", "version", "private_field", "private_metadata", "private_evaluation", "invalid_price", "partial_pair", "precise_coordinate", "unsupported_type"])
def test_store_rejects_corrupt_or_incompatible_artifacts(prepared_explorer, corruption):
    artifact, output, _ = prepared_explorer
    changed = deepcopy(artifact)
    if corruption == "invalid_json":
        output.write_text("{broken", encoding="utf-8")
    else:
        if corruption == "version":
            changed["schema_version"] = 2
        elif corruption == "private_field":
            changed["listings"][0]["email"] = "private@example.test"
        elif corruption == "private_metadata":
            changed["metadata"]["email"] = "private@example.test"
        elif corruption == "private_evaluation":
            changed["evaluation"]["email"] = "private@example.test"
        elif corruption == "invalid_price":
            changed["listings"][0]["price_usd"] = -1
        elif corruption == "partial_pair":
            changed["listings"][0]["lat"] = None
        elif corruption == "precise_coordinate":
            changed["listings"][0]["lat"] = -12.012345
        elif corruption == "unsupported_type":
            changed["listings"][0]["property_type"] = "Oficina"
        output.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(MissingExplorerData) as error:
        ExplorerStore(output).get()
    assert str(output) not in str(error.value)
    assert "private@example.test" not in str(error.value)


def test_missing_store_is_safe_and_recovers_after_generation(tmp_path, synthetic_raw):
    output = tmp_path / "initially-missing.json"
    store = ExplorerStore(output)
    with pytest.raises(MissingExplorerData) as error:
        store.get()
    assert str(output) not in str(error.value)
    source = tmp_path / "fictional.csv"
    synthetic_raw.to_csv(source, index=False)
    prepare_explorer(source, output)
    assert len(store.get()["listings"]) == len(synthetic_raw)


@pytest.mark.parametrize("params", [
    {"currency": "USD"}, {"unknown": "1"}, {"property_type": "Oficina"},
    {"min_price": "-1"}, {"min_price": "nan"}, {"max_price": "inf"},
    {"min_price": "20", "max_price": "10"}, {"min_area": "9"},
    {"max_area": "10001"}, {"min_area": "100", "max_area": "99"},
    {"bedrooms": "1.5"}, {"bathrooms": "-1"}, {"bedrooms": "51"},
    {"limit": "0"}, {"limit": "2001"}, {"limit": "1.5"},
    {"offset": "-1"}, {"offset": str(MAX_EXPLORER_LISTINGS + 1)}, {"q": "x" * 101},
    {"l2": "x" * 101}, {"l4": "x" * 101}, {"l2": ["a", "b"]},
])
def test_invalid_filter_values_are_rejected(params):
    with pytest.raises(FilterValidationError):
        parse_filters(params)


def test_filters_use_exact_counts_inclusive_bounds_and_literal_search(synthetic_raw):
    frame = synthetic_raw.rename(columns={"price": "price_usd"})
    frame.loc[1, "l4"] = "Synthetic [district]"
    row = frame.iloc[1]
    filters = parse_filters({
        "property_type": "Casa", "l2": row["l2"], "l4": row["l4"],
        "min_price": str(row["price_usd"]), "max_price": str(row["price_usd"]),
        "min_area": str(row["surface_total"]), "max_area": str(row["surface_total"]),
        "bedrooms": str(int(row["bedrooms"])), "bathrooms": str(int(row["bathrooms"])),
        "q": "[DISTRICT]",
    })
    selected = filter_listings(frame, filters)
    assert selected.index.tolist() == [1]
    assert filter_listings(frame, parse_filters({"q": ".*"})).empty
    assert len(filter_listings(frame, parse_filters({"q": "synthetic REGION A"}))) == 30
    assert len(filter_listings(frame, parse_filters({"q": "   "}))) == len(frame)
    assert filter_listings(frame, parse_filters({"l2": "Unlisted region"})).empty


def test_exploration_pages_and_api_do_not_require_a_model(prepared_explorer, tmp_path):
    artifact, output, _ = prepared_explorer
    client = create_app(model_path=tmp_path / "missing-model.joblib", explorer_path=output).test_client()
    assert client.get("/explore").status_code == 200
    assert client.get("/insights").status_code == 200
    for endpoint in ("/api/options", "/api/listings", "/api/insights"):
        response = client.get(endpoint)
        assert response.status_code == 200
        json.dumps(response.get_json(), allow_nan=False)
    assert client.get("/api/listings").get_json()["matched_listings"] == len(artifact["listings"])


def test_missing_explorer_keeps_pages_available_with_safe_api_errors(tmp_path):
    missing = tmp_path / "private-missing.json"
    client = create_app(model_path=tmp_path / "missing.joblib", explorer_path=missing).test_client()
    assert client.get("/explore").status_code == 200
    assert client.get("/insights").status_code == 200
    for endpoint in ("/api/options", "/api/listings", "/api/insights"):
        response = client.get(endpoint)
        assert response.status_code == 503
        assert response.get_json()["error"]
        assert str(missing).encode() not in response.data
        assert b"Traceback" not in response.data


@pytest.mark.parametrize("endpoint", ["/api/listings", "/api/insights"])
@pytest.mark.parametrize("query", ["currency=USD", "min_price=nan", "min_area=100&max_area=20", "property_type=Oficina", "l2=a&l2=b"])
def test_api_rejects_invalid_unsupported_and_repeated_filters(prepared_explorer, endpoint, query):
    _, output, _ = prepared_explorer
    response = create_app(explorer_path=output).test_client().get(f"{endpoint}?{query}")
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_dynamic_options_and_unknown_regions(prepared_explorer):
    artifact, output, _ = prepared_explorer
    client = create_app(explorer_path=output).test_client()
    options = client.get("/api/options").get_json()
    assert set(options["property_types"]) == {"Casa", "Departamento"}
    assert set(options["regions"]) == {row["l2"] for row in artifact["listings"]}
    chosen = artifact["listings"][0]["l2"]
    local = client.get("/api/options", query_string={"l2": chosen}).get_json()
    expected = {row["l4"] for row in artifact["listings"] if row["l2"] == chosen}
    assert set(local["localities"]) == expected
    assert client.get("/api/options?l2=Unknown").get_json()["localities"] == []
    assert client.get("/api/options?limit=1").status_code == 400
    assert client.get("/api/options?l2=a&l2=b").status_code == 400


def test_listings_response_is_bounded_and_paginates_without_losing_counts(prepared_explorer):
    artifact, output, _ = prepared_explorer
    client = create_app(explorer_path=output).test_client()
    first = client.get("/api/listings?limit=7").get_json()
    second = client.get("/api/listings?limit=7&offset=7").get_json()
    assert first["matched_listings"] == len(artifact["listings"])
    assert first["returned"] == len(first["listings"]) == 7
    assert first["next_offset"] == 7
    assert first["offset"] == 0 and first["limit"] == 7
    assert second["listings"] == artifact["listings"][7:14]
    assert first["coordinate_precision"] == 2
    assert all(set(row) == LISTING_FIELDS for row in first["listings"])
    last = client.get("/api/listings?limit=7&offset=56").get_json()
    assert last["returned"] == 4 and last["next_offset"] is None
    empty = client.get("/api/listings?l2=Unknown").get_json()
    assert empty["matched_listings"] == empty["mappable_listings"] == empty["returned"] == 0
    assert empty["listings"] == [] and empty["next_offset"] is None


def test_map_excludes_missing_coordinates_while_counts_include_them(tmp_path, synthetic_raw):
    raw = synthetic_raw.copy()
    raw.loc[:4, "lat"] = np.nan
    source, output = tmp_path / "missing-coordinates.csv", tmp_path / "explorer.json"
    raw.to_csv(source, index=False)
    prepare_explorer(source, output)
    client = create_app(explorer_path=output).test_client()
    response = client.get("/api/listings").get_json()
    assert response["matched_listings"] == 60
    assert response["mappable_listings"] == response["returned"] == 55
    assert response["missing_coordinates"] == 5
    assert all(row["lat"] is not None and row["lon"] is not None for row in response["listings"])
    assert client.get("/api/insights").get_json()["summary"]["listings"] == 60


def test_evaluation_uses_verified_saved_test_predictions_and_existing_model(tmp_path, synthetic_evaluation, monkeypatch):
    from sklearn.pipeline import Pipeline

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Evaluation preparation must use the existing fitted model.")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    run = synthetic_evaluation
    output = tmp_path / "verified-explorer.json"
    artifact = prepare_explorer(run["source"], output, run["model"], run["evaluation"], run["predictions"])
    evaluation = artifact["evaluation"]
    assert evaluation["available"] is True
    assert evaluation["held_out_test"] == pytest.approx(run["report"]["held_out_test"])
    assert evaluation["dataset_sha256"] == artifact["metadata"]["dataset_sha256"]
    assert evaluation["evaluation_partition"] == "held-out test"
    assert evaluation["importance_partition"] == "validation"
    assert set(point["feature"] for point in evaluation["importance"]) == set(run["report"]["permutation_importance_validation"])
    assert evaluation["predictions"]["actual_usd"] == pytest.approx(run["actual"].tolist())
    assert evaluation["predictions"]["predicted_usd"] == pytest.approx(run["predicted"].tolist())
    assert evaluation["predictions"]["residual_usd"] == pytest.approx((run["actual"] - run["predicted"]).tolist())
    client = create_app(model_path=run["model"], explorer_path=output).test_client()
    full = client.get("/api/insights").get_json()["evaluation"]
    filtered = client.get("/api/insights?l2=Unknown").get_json()["evaluation"]
    assert filtered == full == evaluation
    assert ExplorerStore(output).get() == artifact


@pytest.mark.parametrize("mismatch", ["report_hash", "model_hash", "model_selection", "actual_order", "predicted_value", "metrics", "count", "nonfinite", "missing_predictions"])
def test_unverified_evaluation_is_disabled_without_losing_listing_exploration(tmp_path, synthetic_evaluation, mismatch):
    run = synthetic_evaluation
    report = deepcopy(run["report"])
    points = pd.read_csv(run["predictions"])
    if mismatch == "report_hash":
        report["dataset_sha256"] = "0" * 64
    elif mismatch == "model_hash":
        metadata = {**run["report"], "dataset_sha256": "0" * 64}
        save_artifact(run["fitted_dummy"], run["model"], metadata)
    elif mismatch == "model_selection":
        metadata = {**run["report"], "selected_model": "random_forest"}
        save_artifact(run["fitted_dummy"], run["model"], metadata)
    elif mismatch == "actual_order":
        points["actual_usd"] = points["actual_usd"].iloc[::-1].to_numpy()
    elif mismatch == "predicted_value":
        points.loc[0, "predicted_usd"] += 1000
    elif mismatch == "metrics":
        report["held_out_test"]["mae_usd"] += 1000
    elif mismatch == "count":
        points = points.iloc[:-1]
    elif mismatch == "nonfinite":
        points.loc[0, "predicted_usd"] = np.inf
    elif mismatch == "missing_predictions":
        run["predictions"] = tmp_path / "missing.csv"
    run["evaluation"].write_text(json.dumps(report), encoding="utf-8")
    if mismatch != "missing_predictions":
        points.to_csv(run["predictions"], index=False)
    output = tmp_path / "safe-explorer.json"
    artifact = prepare_explorer(run["source"], output, run["model"], run["evaluation"], run["predictions"])
    assert len(artifact["listings"]) == 60
    assert artifact["evaluation"]["available"] is False
    assert artifact["evaluation"]["reason"]
    assert "predictions" not in artifact["evaluation"]
    assert ExplorerStore(output).get() == artifact
    json.dumps(artifact, allow_nan=False)


@pytest.mark.parametrize("section", ["comparison", "split", "importance", "metrics"])
def test_evaluation_report_does_not_export_unexpected_private_nested_fields(tmp_path, synthetic_evaluation, section):
    run = synthetic_evaluation
    report = deepcopy(run["report"])
    private = "private@example.test"
    if section == "comparison":
        report["comparison"][0]["email"] = private
    elif section == "split":
        report["split"]["email"] = private
    elif section == "importance":
        report["permutation_importance_validation"]["surface_total"]["email"] = private
    elif section == "metrics":
        report["held_out_test"]["email"] = private
    run["evaluation"].write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "sanitized-evaluation.json"
    artifact = prepare_explorer(run["source"], output, run["model"], run["evaluation"], run["predictions"])
    if section == "split":
        # Unused report metadata may be discarded before the strict artifact contract.
        assert set(artifact["evaluation"]["split"]) == {"rows"}
    else:
        assert artifact["evaluation"]["available"] is False
    assert private not in output.read_text(encoding="utf-8")
    assert len(artifact["listings"]) == 60


@pytest.mark.parametrize("section", ["comparison", "split", "importance", "predictions", "metrics"])
def test_store_rejects_private_fields_added_to_saved_evaluation(tmp_path, synthetic_evaluation, section):
    run = synthetic_evaluation
    output = tmp_path / "corrupted-evaluation.json"
    artifact = prepare_explorer(run["source"], output, run["model"], run["evaluation"], run["predictions"])
    assert artifact["evaluation"]["available"] is True
    evaluation = artifact["evaluation"]
    if section == "comparison":
        evaluation["comparison"][0]["email"] = "private@example.test"
    elif section == "split":
        evaluation["split"]["email"] = "private@example.test"
    elif section == "importance":
        evaluation["importance"][0]["email"] = "private@example.test"
    elif section == "predictions":
        evaluation["predictions"]["email"] = "private@example.test"
    elif section == "metrics":
        evaluation["held_out_test"]["email"] = "private@example.test"
    output.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(MissingExplorerData) as error:
        ExplorerStore(output).get()
    assert "private@example.test" not in str(error.value)


@pytest.mark.parametrize("change", ["train_rows", "validation_rows", "groups", "comparison", "importance", "seed", "run_utc"])
def test_evaluation_rejects_diagnostics_from_a_different_saved_run(tmp_path, synthetic_evaluation, change):
    run = synthetic_evaluation
    report = json.loads(run["evaluation"].read_text(encoding="utf-8"))
    if change == "train_rows":
        report["split"]["rows"]["train"] = 999999
    elif change == "validation_rows":
        report["split"]["rows"]["validation"] = 999999
    elif change == "groups":
        report["split"]["groups"]["train"] = 999999
    elif change == "comparison":
        report["comparison"][0]["validation"]["mae_usd"] = 1.0
    elif change == "importance":
        report["permutation_importance_validation"]["surface_total"]["mean_mae_increase_usd"] = 999999999.0
    elif change == "seed":
        report["seed"] += 1
    elif change == "run_utc":
        report["run_utc"] = "2026-10-06T00:00:00+00:00"
    run["evaluation"].write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    artifact = prepare_explorer(run["source"], tmp_path / "safe-run.json", run["model"], run["evaluation"], run["predictions"])
    assert artifact["evaluation"]["available"] is False
    assert len(artifact["listings"]) == 60


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_evaluation_recomputes_rows_even_when_report_and_metadata_agree(tmp_path, synthetic_evaluation, partition):
    run = synthetic_evaluation
    report = json.loads(run["evaluation"].read_text(encoding="utf-8"))
    report["split"]["rows"][partition] = 999999
    run["evaluation"].write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    save_artifact(run["fitted_dummy"], run["model"], report)
    artifact = prepare_explorer(run["source"], tmp_path / "invalid-counts.json", run["model"], run["evaluation"], run["predictions"])
    assert artifact["evaluation"]["available"] is False


def test_verified_evaluation_remains_compatible_with_missing_optional_model_metadata(tmp_path, synthetic_evaluation):
    run = synthetic_evaluation
    metadata = {"dataset_sha256": run["report"]["dataset_sha256"], "selected_model": run["report"]["selected_model"]}
    save_artifact(run["fitted_dummy"], run["model"], metadata)
    artifact = prepare_explorer(run["source"], tmp_path / "compatible-run.json", run["model"], run["evaluation"], run["predictions"])
    assert artifact["evaluation"]["available"] is True
    assert artifact["evaluation"]["held_out_test"] == pytest.approx(run["report"]["held_out_test"])
    assert artifact["evaluation"]["run_utc"] == run["report"]["run_utc"]


def test_map_pagination_continues_beyond_one_hundred_thousand_records(prepared_explorer):
    artifact, output, _ = prepared_explorer
    base = artifact["listings"][0]
    artifact["listings"] = [{**base, "price_usd": float(100_000 + index)} for index in range(102003)]
    artifact["metadata"]["retained_rows"] = len(artifact["listings"])
    output.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    client = create_app(explorer_path=output).test_client()
    before_boundary = client.get("/api/listings?offset=100000&limit=2000")
    assert before_boundary.status_code == 200
    assert before_boundary.get_json()["next_offset"] == 102000
    after_boundary = client.get("/api/listings?offset=102000&limit=2000")
    assert after_boundary.status_code == 200
    page = after_boundary.get_json()
    assert page["matched_listings"] == page["mappable_listings"] == 102003
    assert page["returned"] == 3 and page["next_offset"] is None
    assert [row["price_usd"] for row in page["listings"]] == [202000.0, 202001.0, 202002.0]
    assert client.get(f"/api/listings?offset={MAX_EXPLORER_LISTINGS}").status_code == 200
    assert client.get(f"/api/listings?offset={MAX_EXPLORER_LISTINGS + 1}").status_code == 400
