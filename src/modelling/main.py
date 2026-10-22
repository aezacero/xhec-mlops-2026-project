# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

import pandas as pd
from prefect import flow, task
from preprocessing import (
    TARGET_COL,
    build_features,
    clean_column_names,
    fit_encoder,
    load_data,
    remove_outliers,
)
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from training import evaluate_model, train_model
from utils import pickle_object

TEST_SIZE = 0.2
RANDOM_STATE = 42

LOCAL_OBJECTS_DIR = Path(__file__).resolve().parents[1] / "web_service" / "local_objects"
ENCODER_PATH = LOCAL_OBJECTS_DIR / "encoder.pkl"
MODEL_PATH = LOCAL_OBJECTS_DIR / "model.pkl"


@task
def prepare_data(trainset_path: Path) -> pd.DataFrame:
    """Load the dataset, clean the column names and drop the outliers."""
    return remove_outliers(clean_column_names(load_data(trainset_path)))


@task
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into a training and a test set."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)


@task
def fit_and_encode(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[OneHotEncoder, pd.DataFrame, pd.DataFrame]:
    """Fit the encoder on the training set and build the features of both sets."""
    # The encoder is fitted on the training split only, so the test split stays unseen
    encoder = fit_encoder(train_df)
    return encoder, build_features(train_df, encoder), build_features(test_df, encoder)


@task
def train_and_evaluate(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> RegressorMixin:
    """Train the model, print its test metrics next to a mean baseline and return it."""
    model = train_model(LinearRegression(), X_train, y_train)
    baseline = train_model(DummyRegressor(strategy="mean"), X_train, y_train)
    for name, fitted in [("baseline_mean", baseline), ("linear_regression", model)]:
        metrics = evaluate_model(fitted, X_test, y_test)
        print(f"{name:<18} " + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items()))
    return model


@task
def save_objects(encoder: OneHotEncoder, model: RegressorMixin) -> None:
    """Pickle the fitted encoder and model for the API."""
    pickle_object(encoder, ENCODER_PATH)
    pickle_object(model, MODEL_PATH)
    print(f"Saved the encoder and model to {LOCAL_OBJECTS_DIR}")


@flow(log_prints=True)
def train_model_workflow(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model and encoder (pickle)."""
    df = prepare_data(trainset_path)
    train_df, test_df = split_data(df)
    encoder, X_train, X_test = fit_and_encode(train_df, test_df)
    model = train_and_evaluate(X_train, train_df[TARGET_COL], X_test, test_df[TARGET_COL])
    save_objects(encoder, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=Path, help="Path to the training set")
    args = parser.parse_args()
    train_model_workflow(args.trainset_path)
