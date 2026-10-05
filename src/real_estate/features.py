"""Deterministic feature engineering shared by training and serving."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .config import FEATURE_COLUMNS


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Use coarse Peru geography; no unsupported area or room ratios."""

    def fit(self, X: pd.DataFrame, y=None):
        self._check(X)
        self.feature_names_in_ = np.array(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        return self

    @staticmethod
    def _check(X):
        if not isinstance(X, pd.DataFrame) or set(FEATURE_COLUMNS) - set(X.columns):
            raise ValueError("Expected a DataFrame containing the canonical raw feature columns.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        self._check(X)
        out = X.loc[:, FEATURE_COLUMNS].copy()
        out[["lat", "lon"]] = out[["lat", "lon"]].round(2)
        lat, lon = np.radians(out["lat"]), np.radians(out["lon"])
        ref_lat, ref_lon = np.radians([-12.0464, -77.0428])
        a = np.sin((lat - ref_lat) / 2) ** 2 + np.cos(ref_lat) * np.cos(lat) * np.sin((lon - ref_lon) / 2) ** 2
        out["distance_to_lima_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([*FEATURE_COLUMNS, "distance_to_lima_km"], dtype=object)
