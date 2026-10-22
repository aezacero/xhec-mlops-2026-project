import numpy as np
import pandas as pd
from preprocessing import build_features, clean_column_names
from sklearn.base import RegressorMixin
from sklearn.preprocessing import OneHotEncoder

# A linear model can predict fewer than 1 ring for the smallest infants
MIN_RINGS = 1.0


def predict(df: pd.DataFrame, encoder: OneHotEncoder, model: RegressorMixin) -> np.ndarray:
    """Predict the number of rings of each abalone, clipped to at least MIN_RINGS."""
    df = clean_column_names(df)
    X = build_features(df, encoder)
    return np.clip(model.predict(X), MIN_RINGS, None)
