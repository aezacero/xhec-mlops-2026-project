from pathlib import Path

from fastapi import FastAPI
from utils import load_pickle

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import InputData, PredictionOut

LOCAL_OBJECTS_DIR = Path(__file__).resolve().parent / "local_objects"

# Loaded once at startup: restart the API to use a newly trained model
encoder = load_pickle(LOCAL_OBJECTS_DIR / "encoder.pkl")
model = load_pickle(LOCAL_OBJECTS_DIR / "model.pkl")

app = FastAPI(
    title="Abalone age prediction",
    description="Predict the number of rings of an abalone from its physical measurements.",
)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData) -> dict:
    return {"rings": run_inference(payload, encoder, model)}
