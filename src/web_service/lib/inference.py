import pandas as pd
from predicting import predict
from sklearn.base import RegressorMixin
from sklearn.preprocessing import OneHotEncoder

from src.web_service.lib.models import InputData


def run_inference(input_data: InputData, encoder: OneHotEncoder, model: RegressorMixin) -> float:
    """Predict the number of rings of one abalone with the saved encoder and model."""
    df = pd.DataFrame([input_data.model_dump()])
    return float(predict(df, encoder, model)[0])
