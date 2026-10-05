"""Strict public input validation; optional missing values remain missing."""

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_COLUMNS, COUNT_COLUMNS, FEATURE_COLUMNS, NUMERIC_BOUNDS,
    NUMERIC_COLUMNS, PROPERTY_TYPES,
)


class InputValidationError(ValueError):
    """A safe, user-readable input error."""


def validate_payload(payload: Mapping) -> pd.DataFrame:
    if not isinstance(payload, Mapping):
        raise InputValidationError("Provide a JSON object or a property form.")
    if set(payload) - set(FEATURE_COLUMNS):
        raise InputValidationError("Input contains unsupported fields.")
    row = {}
    for name in NUMERIC_COLUMNS:
        value = payload.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            if name == "surface_total":
                raise InputValidationError("surface_total is required.")
            row[name] = np.nan
            continue
        if isinstance(value, (bool, np.bool_, list, dict)):
            raise InputValidationError(f"{name} must be a number.")
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise InputValidationError(f"{name} must be a number.") from exc
        lower, upper = NUMERIC_BOUNDS[name]
        if not math.isfinite(number) or not lower <= number <= upper:
            raise InputValidationError(f"{name} must be between {lower} and {upper}.")
        if name in COUNT_COLUMNS and not number.is_integer():
            raise InputValidationError(f"{name} must be a whole number.")
        row[name] = number
    if math.isnan(row["lat"]) != math.isnan(row["lon"]):
        raise InputValidationError("Provide both latitude and longitude, or leave both blank.")
    for name in CATEGORICAL_COLUMNS:
        value = payload.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            if name == "property_type":
                raise InputValidationError("property_type is required.")
            row[name] = np.nan
            continue
        if not isinstance(value, str) or len(value.strip()) > 100 or any(ord(c) < 32 for c in value):
            raise InputValidationError(f"{name} must be text of at most 100 characters.")
        row[name] = value.strip()
    if row["property_type"] not in PROPERTY_TYPES:
        raise InputValidationError("property_type must be Casa or Departamento.")
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
