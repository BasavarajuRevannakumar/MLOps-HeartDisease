from fastapi import FastAPI

from src.api.predictor import predict
from src.api.schemas import (
    HeartDiseaseRequest,
    PredictionResponse,
)
from src.api.middleware import log_requests
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
)
app.middleware("http")(log_requests)
Instrumentator().instrument(app).expose(app)

@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessor_loaded": True
    }

@app.get("/version")
def version():
    return {
        "model": "Random Forest",
        "version": "1.0.0"
    }

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_endpoint(request: HeartDiseaseRequest):

    result = predict(request.model_dump())

    return result