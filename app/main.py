from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model import load_artifacts, predict
from app.schemas import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Wine Classification ML API",
    description="FastAPI service for predicting wine class using a trained scikit-learn model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    load_artifacts()
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest) -> PredictionResponse:
    result = predict(request.model_dump())
    return PredictionResponse(**result)
