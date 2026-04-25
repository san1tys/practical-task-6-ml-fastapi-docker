from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    fixed_acidity: float = Field(..., gt=0)
    volatile_acidity: float = Field(..., gt=0)
    citric_acid: float = Field(..., ge=0)
    residual_sugar: float = Field(..., gt=0)
    chlorides: float = Field(..., gt=0)
    free_sulfur_dioxide: float = Field(..., gt=0)
    total_sulfur_dioxide: float = Field(..., gt=0)
    density: float = Field(..., gt=0)
    pH: float = Field(..., gt=0)
    sulphates: float = Field(..., gt=0)
    alcohol: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
