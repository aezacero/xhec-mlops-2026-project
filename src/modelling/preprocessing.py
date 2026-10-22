from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = ["sex"]
NUMERICAL_COLS = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
TARGET_COL = "rings"


def load_data(path: Path) -> pd.DataFrame:
    """Read the abalone dataset from a CSV file."""
    return pd.read_csv(path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the column names and replace spaces with underscores."""
    return df.rename(columns=lambda col: col.strip().lower().replace(" ", "_"))


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows with an impossible (0) or extreme (> 0.4) height."""
    return df[(df["height"] > 0) & (df["height"] <= 0.4)]


def fit_encoder(df: pd.DataFrame) -> OneHotEncoder:
    """Fit a one-hot encoder on the categorical columns."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.set_output(transform="pandas")
    return encoder.fit(df[CATEGORICAL_COLS])


def build_features(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """Build the feature matrix: numerical columns followed by the encoded categorical ones."""
    encoded = encoder.transform(df[CATEGORICAL_COLS])
    return pd.concat([df[NUMERICAL_COLS], encoded], axis=1)
