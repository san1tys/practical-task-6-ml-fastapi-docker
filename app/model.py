from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "features.joblib"
TARGET_NAMES_PATH = ARTIFACTS_DIR / "target_names.joblib"

_model = None
_features = None
_target_names = None


def load_artifacts() -> tuple:
    global _model, _features, _target_names
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _features = joblib.load(FEATURES_PATH)
        _target_names = joblib.load(TARGET_NAMES_PATH)
    return _model, _features, _target_names


def predict(payload: dict) -> dict:
    model, features, target_names = load_artifacts()
    frame = pd.DataFrame([payload], columns=features)
    prediction = int(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]

    return {
        "predicted_class": prediction,
        "predicted_label": str(target_names[prediction]),
        "probabilities": {
            str(target_names[index]): round(float(prob), 6)
            for index, prob in enumerate(probabilities)
        },
    }
