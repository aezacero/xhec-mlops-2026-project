import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def train_model(model: RegressorMixin, X: pd.DataFrame, y: pd.Series) -> RegressorMixin:
    """Fit a scikit-learn regressor on the given features and target."""
    return model.fit(X, y)


def evaluate_model(model: RegressorMixin, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute the RMSE, MAE and R² of the model's predictions."""
    y_pred = model.predict(X)
    return {
        "rmse": root_mean_squared_error(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred),
    }
