import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from ml.config import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURE_COLUMNS, EXCLUDED_FEATURES
)

def create_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer:
    - Categorical features (proto, service, state) one-hot encoded with handle_unknown='ignore'
    - Numeric features scaled with StandardScaler
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor

def prepare_features(df: pd.DataFrame, is_training: bool = False):
    """
    Extracts features X and target y (if present).
    Ensures id, attack_cat, and label are excluded from model inputs.
    """
    # Exclude target and metadata
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    # Fill missing values if any
    X = df[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("-").astype(str)
            
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    y = None
    if "label" in df.columns:
        y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values

    return X, y

def downcast_matrix_to_float32(X_matrix: np.ndarray) -> np.ndarray:
    """Downcasts float64 matrix to float32 to optimize memory."""
    if X_matrix.dtype == np.float64:
        return X_matrix.astype(np.float32)
    return X_matrix
