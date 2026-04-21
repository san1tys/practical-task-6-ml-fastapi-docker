from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine
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


def main() -> None:
    dataset = load_wine(as_frame=True)
    frame = dataset.frame.copy()

    # Save a local CSV so the project includes an explicit dataset file.
    csv_columns = [column.replace("/", "_").replace(" ", "_") for column in frame.columns]
    frame.columns = csv_columns
    dataset_csv_path = DATA_DIR / "wine_dataset.csv"
    frame.to_csv(dataset_csv_path, index=False)

    features = [column for column in frame.columns if column != "target"]
    target = "target"

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
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=dataset.target_names, output_dict=True)

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(features, ARTIFACTS_DIR / "features.joblib")
    joblib.dump(list(dataset.target_names), ARTIFACTS_DIR / "target_names.joblib")

    metrics = {
        "dataset": "sklearn wine dataset",
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "feature_count": len(features),
        "test_accuracy": round(float(accuracy), 6),
        "classification_report": report,
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("wine-classification")

    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": 250,
            "max_depth": 8,
            "random_state": 42,
            "test_size": 0.2,
            "scaler": "StandardScaler",
        })
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })
        for class_name in dataset.target_names:
            key = str(class_name)
            mlflow.log_metric(f"{key}_precision", report[key]["precision"])
            mlflow.log_metric(f"{key}_recall",    report[key]["recall"])
            mlflow.log_metric(f"{key}_f1",        report[key]["f1-score"])
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="wine-classifier",
        )

    print("Training complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
