"""One validated filtering contract for map and analytical views."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .config import PROPERTY_TYPES

MAX_EXPLORER_LISTINGS = 250_000


class FilterValidationError(ValueError):
    """A safe filter error suitable for the user interface."""


@dataclass(frozen=True)
class ListingFilters:
    property_type: str | None = None
    l2: str | None = None
    l4: str | None = None
    q: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_area: float | None = None
    max_area: float | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    limit: int = 2000
    offset: int = 0


def parse_filters(values: Mapping) -> ListingFilters:
    allowed = set(ListingFilters.__dataclass_fields__)
    if not isinstance(values, Mapping) or set(values) - allowed:
        raise FilterValidationError("Unsupported filter parameter.")
    parsed = {}
    for name, value in values.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if name in {"property_type", "l2", "l4", "q"}:
            if not isinstance(value, str) or len(value.strip()) > 100 or any(ord(c) < 32 for c in value):
                raise FilterValidationError(f"{name} must be text of at most 100 characters.")
            parsed[name] = value.strip()
            if name == "property_type" and parsed[name] not in PROPERTY_TYPES:
                raise FilterValidationError("property_type must be Casa or Departamento.")
            continue
        if isinstance(value, (bool, np.bool_, list, dict)):
            raise FilterValidationError(f"{name} must be a finite number.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FilterValidationError(f"{name} must be a finite number.") from exc
        if not math.isfinite(number):
            raise FilterValidationError(f"{name} must be a finite number.")
        if name in {"bedrooms", "bathrooms", "limit", "offset"}:
            lower, upper = {"bedrooms": (0, 50), "bathrooms": (0, 50), "limit": (1, 2000), "offset": (0, MAX_EXPLORER_LISTINGS)}[name]
            if not number.is_integer() or not lower <= number <= upper:
                raise FilterValidationError(f"{name} must be a whole number between {lower} and {upper}.")
            parsed[name] = int(number)
        else:
            lower, upper = (10, 10000) if name.endswith("area") else (0, math.inf)
            if not lower <= number <= upper:
                raise FilterValidationError(f"{name} must be between {lower} and {upper}.")
            parsed[name] = number
    for quantity in ("price", "area"):
        lower, upper = parsed.get(f"min_{quantity}"), parsed.get(f"max_{quantity}")
        if lower is not None and upper is not None and lower > upper:
            raise FilterValidationError(f"Minimum {quantity} cannot exceed maximum {quantity}.")
    return ListingFilters(**parsed)


def filter_listings(frame: pd.DataFrame, filters: ListingFilters) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for name in ("property_type", "l2", "l4", "bedrooms", "bathrooms"):
        value = getattr(filters, name)
        if value is not None:
            mask &= frame[name].eq(value)
    for suffix, column in (("price", "price_usd"), ("area", "surface_total")):
        lower, upper = getattr(filters, f"min_{suffix}"), getattr(filters, f"max_{suffix}")
        if lower is not None:
            mask &= frame[column].ge(lower)
        if upper is not None:
            mask &= frame[column].le(upper)
    if filters.q:
        mask &= (frame["l2"].fillna("") + " " + frame["l4"].fillna("")).str.casefold().str.contains(
            filters.q.casefold(), regex=False)
    return frame.loc[mask].copy()
