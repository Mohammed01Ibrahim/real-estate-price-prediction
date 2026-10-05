"""Backend-owned aggregate statistics from the corrected listing scope."""

import numpy as np
import pandas as pd

from .explorer import LISTING_FIELDS, _json_rows
from .filters import ListingFilters, filter_listings

CORRELATION_COLUMNS = ("surface_total", "bedrooms", "bathrooms", "lat", "lon", "price_usd")


def _frame(artifact):
    return pd.DataFrame(artifact["listings"], columns=LISTING_FIELDS)


def histogram(values, logarithmic=False):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"edges": [], "counts": []}
    lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        lower = lower / 1.05 if logarithmic else lower - max(abs(lower) * .05, .5)
        upper = upper * 1.05 if logarithmic else upper + max(abs(upper) * .05, .5)
    edges = np.geomspace(lower, upper, 41) if logarithmic else np.linspace(lower, upper, 41)
    counts, edges = np.histogram(values, bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def listings_response(artifact: dict, filters: ListingFilters) -> dict:
    filtered = filter_listings(_frame(artifact), filters)
    located = filtered.loc[filtered["lat"].notna() & filtered["lon"].notna()]
    page = located.iloc[filters.offset:filters.offset + filters.limit]
    end = filters.offset + len(page)
    return {"listings": _json_rows(page), "matched_listings": len(filtered), "mappable_listings": len(located),
            "missing_coordinates": len(filtered) - len(located), "returned": len(page), "offset": filters.offset,
            "limit": filters.limit, "next_offset": end if end < len(located) else None, "coordinate_precision": 2}


def _scatter(frame, maximum=2000):
    count = len(frame)
    if count > maximum:
        # Deterministic sample includes range endpoints; distributions still use every row.
        extremes = set()
        for column in ("price_usd", "surface_total"):
            extremes.update((frame[column].idxmin(), frame[column].idxmax()))
        remaining = frame.drop(index=list(extremes))
        selected = pd.concat([frame.loc[sorted(extremes)], remaining.sample(n=maximum - len(extremes), random_state=42)])
    else:
        selected = frame
    columns = ["price_usd", "surface_total", "property_type", "l2", "l4"]
    return {"points": _json_rows(selected.loc[:, columns]), "total": count, "shown": len(selected), "sampled": count > maximum}


def insights_response(artifact: dict, filters: ListingFilters) -> dict:
    frame = filter_listings(_frame(artifact), filters)
    count = len(frame)
    summary = {"listings": count, "regions": int(frame["l2"].nunique()),
               "median_price_usd": float(frame["price_usd"].median()) if count else None,
               "median_area_m2": float(frame["surface_total"].median()) if count else None,
               "casa": int(frame["property_type"].eq("Casa").sum()),
               "departamento": int(frame["property_type"].eq("Departamento").sum())}
    price = histogram(frame["price_usd"])
    log_price = histogram(frame["price_usd"], logarithmic=True)
    price.update({"log_edges": log_price["edges"], "log_counts": log_price["counts"]})
    regional = frame.assign(region=frame["l2"].fillna("Unknown")).groupby("region")["price_usd"].agg(["count", "median"])
    regions = [{"region": str(name), "count": int(row["count"]), "median_price_usd": float(row["median"])}
               for name, row in regional.sort_values("count", ascending=False).iterrows()]
    numeric = frame.loc[:, CORRELATION_COLUMNS].astype(float)
    corr = numeric.corr(min_periods=2).reindex(index=CORRELATION_COLUMNS, columns=CORRELATION_COLUMNS)
    pairs = numeric.notna().astype(int).T.dot(numeric.notna().astype(int))
    matrix = [[None if pd.isna(value) else float(value) for value in row] for row in corr.to_numpy()]
    return {
        "summary": summary, "price_distribution": price, "area_distribution": histogram(frame["surface_total"]),
        "property_types": [{"property_type": name, "count": int(frame["property_type"].eq(name).sum())}
                           for name in ("Casa", "Departamento")],
        "regions": regions, "scatter": _scatter(frame),
        "correlation": {"variables": list(CORRELATION_COLUMNS), "matrix": matrix,
                        "pair_counts": pairs.to_numpy().tolist()},
        "evaluation": artifact["evaluation"],
    }


def filter_options(artifact, region=None):
    frame = _frame(artifact)
    localities = frame if region is None else frame.loc[frame["l2"].eq(region)]
    return {"property_types": ["Casa", "Departamento"], "regions": sorted(frame["l2"].dropna().unique().tolist()),
            "localities": sorted(localities["l4"].dropna().unique().tolist())}
