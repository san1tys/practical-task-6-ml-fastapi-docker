from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

# Cosmetic: integer class-label outputs trigger an MLflow schema hint that
# does not apply to classifiers (predictions are never missing).
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column",
)

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


_QUALITY_BIN: dict[int, int] = {3: 0, 4: 0, 5: 0, 6: 1, 7: 2, 8: 2}
_TARGET_NAMES = ["quality_low", "quality_medium", "quality_high"]


def main() -> None:
    frame = pd.read_csv(DATA_DIR / "WineQT.csv")
    frame = frame.drop(columns=["Id"])
    frame.columns = [col.replace(" ", "_") for col in frame.columns]

    # Merge rare extreme classes: 3+4 → "low" (0), 7+8 → "high" (3).
    # quality_3 has ~5 samples and quality_8 ~15 — too few to learn separately.
    frame["quality_bin"] = frame["quality"].map(_QUALITY_BIN)

    features = [col for col in frame.columns if col not in ("quality", "quality_bin")]
    target = "quality_bin"

    classes = [0, 1, 2]
    target_names = _TARGET_NAMES

    X_train, X_test, y_train, y_test = train_test_split(
        frame[features],
        frame[target],
        test_size=0.2,
        random_state=42,
        stratify=frame[target],
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, labels=classes, target_names=target_names, output_dict=True
    )

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(features, ARTIFACTS_DIR / "features.joblib")
    joblib.dump(target_names, ARTIFACTS_DIR / "target_names.joblib")
    joblib.dump(classes, ARTIFACTS_DIR / "classes.joblib")

    metrics = {
        "dataset": "WineQT",
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "feature_count": len(features),
        "test_accuracy": round(float(accuracy), 6),
        "classification_report": report,
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    default_tracking_uri = (BASE_DIR / "mlruns").as_uri()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", default_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("wine-quality")

    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": 250,
            "max_depth": 8,
            "random_state": 42,
            "test_size": 0.2,
            "scaler": "StandardScaler",
            "class_weight": "balanced",
            "quality_bins": "3-5=low,6=medium,7-8=high",
        })
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })
        for name in target_names:
            mlflow.log_metric(f"{name}_precision", report[name]["precision"])
            mlflow.log_metric(f"{name}_recall",    report[name]["recall"])
            mlflow.log_metric(f"{name}_f1",        report[name]["f1-score"])
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="wine-quality-classifier",
            input_example=X_train.iloc[:1].astype("float64"),
        )

    print("Training complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
