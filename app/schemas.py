from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    alcohol: float = Field(..., gt=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., gt=0)
    alcalinity_of_ash: float = Field(..., gt=0)
    magnesium: float = Field(..., gt=0)
    total_phenols: float = Field(..., gt=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., gt=0)
    od280_od315_of_diluted_wines: float = Field(..., gt=0)
    proline: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
