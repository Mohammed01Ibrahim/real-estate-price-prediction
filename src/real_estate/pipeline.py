"""Every fitted transformation lives inside the estimator evaluated by CV."""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, ENGINEERED_NUMERIC_COLUMNS, SEED
from .features import FeatureEngineer

MODEL_NAMES = ("dummy", "ridge", "hist_gradient_boosting", "random_forest")


def build_pipeline(model_name: str = "hist_gradient_boosting", seed: int = SEED):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, list(ENGINEERED_NUMERIC_COLUMNS)),
        ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
    ], sparse_threshold=0)
    models = {
        "dummy": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=10),
        # No internal random holdout that could separate repeated-listing groups.
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=2, early_stopping=False, random_state=seed,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_leaf=4,
            max_features=0.8, n_jobs=2, random_state=seed,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}.")
    pipeline = Pipeline([
        ("features", FeatureEngineer()), ("preprocess", preprocessing), ("regressor", models[model_name]),
    ])
    if model_name == "dummy":
        return pipeline
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
