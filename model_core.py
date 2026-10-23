"""Shared preprocessing and model-artifact helpers for the training and API code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FORMAT_VERSION = 1


def prepare_features(
    features: pd.DataFrame, numeric_columns: list[str]
) -> pd.DataFrame:
    """Normalize numeric and categorical values to the training schema."""
    cleaned = features.copy()
    numeric_set = set(numeric_columns)
    missing = numeric_set - set(cleaned.columns)
    if missing:
        raise ValueError("Missing numeric feature(s): " + ", ".join(sorted(missing)))
    for column in numeric_columns:
        values = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = values.replace([np.inf, -np.inf], np.nan)
    for column in cleaned.columns:
        if column not in numeric_set:
            values = cleaned[column].astype("string").astype(object)
            cleaned[column] = values.where(pd.notna(values), np.nan)
    return cleaned


def build_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transforms: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transforms.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transforms.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transforms, remainder="drop")),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )


def load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format.")
    required = {"pipeline", "feature_columns", "numeric_columns", "target_column"}
    if not required.issubset(bundle):
        raise ValueError("Model artifact is missing required metadata.")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ValueError("Model artifact does not support probability predictions.")
    if not isinstance(bundle["feature_columns"], list) or not bundle["feature_columns"]:
        raise ValueError("Model artifact has no feature schema.")
    if not set(bundle["numeric_columns"]).issubset(bundle["feature_columns"]):
        raise ValueError("Model artifact numeric features do not match its schema.")
    return bundle
