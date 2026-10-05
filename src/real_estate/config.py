"""Portable defaults and the public feature contract."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pe_properties_crude.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "pipeline.joblib"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
SEED = 42
PROPERTY_TYPES = ("Casa", "Departamento")
FEATURE_COLUMNS = (
    "surface_total", "lat", "lon", "bedrooms",
    "bathrooms", "property_type", "l2", "l4",
)
NUMERIC_COLUMNS = FEATURE_COLUMNS[:5]
CATEGORICAL_COLUMNS = FEATURE_COLUMNS[5:]
NUMERIC_BOUNDS = {
    "surface_total": (10, 10000),
    "lat": (-19, 1), "lon": (-82, -68),
    "bedrooms": (0, 50), "bathrooms": (0, 50),
}
COUNT_COLUMNS = ("bedrooms", "bathrooms")
ENGINEERED_NUMERIC_COLUMNS = NUMERIC_COLUMNS + (
    "distance_to_lima_km",
)
