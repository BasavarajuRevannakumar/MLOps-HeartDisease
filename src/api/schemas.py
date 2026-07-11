from pydantic import BaseModel


class HeartDiseaseRequest(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float