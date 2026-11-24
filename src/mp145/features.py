"""Feature columns shared by the API and the prediction scripts.

The order matches params.yaml: numeric columns, then categorical columns.
`gb_from_monthly_bill` is excluded on purpose: it is filled in after the event
(leakage) and is absent from holdout_features.csv.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "age", "screen_hours", "apps_installed",
    "plan_gb_limit", "video_minutes", "wifi_hours",
]
CATEGORICAL_FEATURES = ["phone_tier", "circle"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the columns the model is allowed to see."""
    return df[FEATURE_COLUMNS].copy()


def build_preprocessor() -> ColumnTransformer:
    """Preprocessing used by the API's offline fallback model."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES),
         ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def make_request_frame(payload: dict) -> pd.DataFrame:
    """Turn a validated request body into the one-row frame the model expects."""
    return pd.DataFrame([payload])[FEATURE_COLUMNS]
