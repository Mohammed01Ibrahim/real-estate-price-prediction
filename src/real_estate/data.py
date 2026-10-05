"""Explicit, deterministic scope filtering without learned statistics."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_COLUMNS, COUNT_COLUMNS, DATA_PATH, FEATURE_COLUMNS,
    NUMERIC_BOUNDS, NUMERIC_COLUMNS, PROPERTY_TYPES,
)

RAW_COLUMNS = ("id", "operation_type", "currency", "price_period", "l1", "price", "surface_covered", "rooms") + FEATURE_COLUMNS


def read_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read only the required schema; never load contact text or addresses."""
    return pd.read_csv(path, usecols=list(RAW_COLUMNS), low_memory=False)


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {', '.join(sorted(missing))}")
    df = raw.loc[:, RAW_COLUMNS].copy()
    report = {"raw_rows": len(df), "sequential_exclusions": {}}
    rules = (
        ("not_peru", df["l1"].eq("Perú")),
        ("not_sale", df["operation_type"].eq("Venta")),
        ("not_usd", df["currency"].eq("USD")),
        ("nonblank_payment_period", df["price_period"].fillna("").astype(str).str.strip().eq("")),
        ("unsupported_property_type", df["property_type"].isin(PROPERTY_TYPES)),
    )
    mask = pd.Series(True, index=df.index)
    for reason, valid in rules:
        report["sequential_exclusions"][reason] = int((mask & ~valid).sum())
        mask &= valid
    df = df.loc[mask].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    valid_price = np.isfinite(df["price"]) & df["price"].gt(0)
    report["sequential_exclusions"]["missing_or_invalid_price"] = int((~valid_price).sum())
    df = df.loc[valid_price].copy()
    report["invalid_optional_cells_to_missing"] = {}
    for name in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[name], errors="coerce")
        lower, upper = NUMERIC_BOUNDS[name]
        boolean_values = df[name].map(lambda v: isinstance(v, (bool, np.bool_)))
        valid = np.isfinite(values) & values.between(lower, upper) & ~boolean_values
        if name in COUNT_COLUMNS:
            valid &= values.mod(1).eq(0)
        if name == "surface_total":
            df[name] = values.where(valid)
        else:
            report["invalid_optional_cells_to_missing"][name] = int((df[name].notna() & ~valid).sum())
            df[name] = values.where(valid)
    valid_area = df["surface_total"].notna()
    report["sequential_exclusions"]["missing_or_out_of_scope_total_area"] = int((~valid_area).sum())
    df = df.loc[valid_area].copy()
    partial_coordinates = df["lat"].isna() | df["lon"].isna()
    report["coordinate_pairs_missing"] = int(partial_coordinates.sum())
    df.loc[partial_coordinates, ["lat", "lon"]] = np.nan
    for name in CATEGORICAL_COLUMNS:
        df[name] = df[name].map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else np.nan)
    keys = df.loc[:, FEATURE_COLUMNS].copy()
    keys[["lat", "lon"]] = keys[["lat", "lon"]].round(2)
    df["_group"] = pd.util.hash_pandas_object(keys, index=False)
    present_id = df["id"].notna() & df["id"].astype(str).str.strip().ne("")
    if df.loc[present_id].groupby("id")["_group"].nunique().gt(1).any():
        raise ValueError("Repeated listing IDs have conflicting feature groups; resolve repeated snapshots before training.")
    before = len(df)
    df = df.drop_duplicates(subset=[*FEATURE_COLUMNS, "price"]).reset_index(drop=True)
    report["sequential_exclusions"]["duplicate_feature_price_observations"] = before - len(df)
    report["unused_source_fields"] = {
        name: {"missing_rows": int(df[name].isna().sum()), "reason": "Not used by the supported predictor contract"}
        for name in ("rooms", "surface_covered")
    }
    X = df.loc[:, FEATURE_COLUMNS].copy()
    y = df["price"].copy()
    # A proxy for repeated listings, independent of price. Not an entity-resolution guarantee.
    groups = df["_group"].rename("group")
    report.update({"retained_rows": len(X), "unique_groups": int(groups.nunique()),
                   "price_min_usd": float(y.min()) if len(y) else None,
                   "price_max_usd": float(y.max()) if len(y) else None,
                   "missing_features": {k: int(v) for k, v in X.isna().sum().items()}})
    return X, y, groups, report
