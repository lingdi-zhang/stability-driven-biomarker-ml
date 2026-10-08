"""Input checks shared by discovery and holdout evaluation."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def validate_features(X, y, name="X"):
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError(f"{name} must be a nonempty pandas DataFrame")
    if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
        raise ValueError(f"{name} must have unique string feature names")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError(f"{name} features must be numeric")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} cannot contain missing or infinite values")
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(X) or pd.isna(labels).any():
        raise ValueError(f"{name} labels must be one-dimensional, nonmissing, and match its rows")
    return labels


def fit_binary_encoder(y):
    encoder = LabelEncoder().fit(y)
    if len(encoder.classes_) != 2:
        raise ValueError("Exactly two target classes are required")
    return encoder, encoder.transform(y)


def validate_cv_counts(y, nested=False):
    minimum = 7 if nested else 5
    if np.bincount(np.asarray(y, dtype=int), minlength=2).min() < minimum:
        raise ValueError(f"Each class needs at least {minimum} samples for {'nested ' if nested else ''}5-fold CV")


def validate_fraction(value, name):
    if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
