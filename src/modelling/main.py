# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from preprocessing import (
    TARGET_COL,
    build_features,
    clean_column_names,
    fit_encoder,
    load_data,
    remove_outliers,
)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from training import evaluate_model, train_model
from utils import pickle_object

TEST_SIZE = 0.2
RANDOM_STATE = 42

LOCAL_OBJECTS_DIR = Path(__file__).resolve().parents[1] / "web_service" / "local_objects"
ENCODER_PATH = LOCAL_OBJECTS_DIR / "encoder.pkl"
MODEL_PATH = LOCAL_OBJECTS_DIR / "model.pkl"


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model and encoder (pickle)."""
    df = remove_outliers(clean_column_names(load_data(trainset_path)))
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # The encoder is fitted on the training split only, so the test split stays unseen
    encoder = fit_encoder(train_df)
    X_train, y_train = build_features(train_df, encoder), train_df[TARGET_COL]
    X_test, y_test = build_features(test_df, encoder), test_df[TARGET_COL]

    model = train_model(LinearRegression(), X_train, y_train)
    baseline = train_model(DummyRegressor(strategy="mean"), X_train, y_train)
    for name, fitted in [("baseline_mean", baseline), ("linear_regression", model)]:
        metrics = evaluate_model(fitted, X_test, y_test)
        print(f"{name:<18} " + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items()))

    pickle_object(encoder, ENCODER_PATH)
    pickle_object(model, MODEL_PATH)
    print(f"Saved the encoder and model to {LOCAL_OBJECTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=Path, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
