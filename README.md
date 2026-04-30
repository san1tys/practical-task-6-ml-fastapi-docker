# Wine Quality Classification — FastAPI, Docker, Streamlit, MLflow

This repository contains a machine learning deployment project for wine quality classification.
It covers two connected assignments:

- **Practical Task 6** — train a machine learning model, save it, and deploy it as an API using FastAPI and Docker.
- **SIS-3 continuation** — extend the project with a Streamlit frontend and integrate MLflow for experiment tracking and model registration.

The project uses the **WineQT** dataset and classifies wines into three quality groups:

- `quality_low`
- `quality_medium`
- `quality_high`

## Technologies

- Python 3.11
- pandas, NumPy
- scikit-learn (`Pipeline`, `StandardScaler`, `RandomForestClassifier`)
- FastAPI, Uvicorn, Pydantic
- joblib
- Streamlit
- MLflow
- Docker and Docker Compose
- pytest

## Project structure

```text
practical-task-6-ml/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application and API endpoints
│   ├── model.py           # model artifact loading and prediction logic
│   └── schemas.py         # Pydantic request/response schemas
├── artifacts/
│   ├── model.joblib       # trained scikit-learn pipeline
│   ├── features.joblib    # ordered list of input features
│   ├── target_names.joblib # class labels
│   ├── classes.joblib     # numeric class IDs
│   └── metrics.json       # evaluation results
├── data/
│   └── WineQT.csv         # WineQT dataset
├── frontend/
│   ├── app.py             # Streamlit frontend
│   ├── requirements.txt   # frontend dependencies
│   └── Dockerfile         # Streamlit Docker image
├── tests/
│   ├── conftest.py
│   └── test_smoke.py      # smoke tests for API endpoints
├── train.py               # training script + MLflow logging
├── requirements.txt       # backend/training dependencies
├── Dockerfile             # FastAPI Docker image
├── docker-compose.yml     # MLflow + API + Streamlit services
├── .dockerignore
└── README.md
```

## Dataset

The project uses the local dataset:

```text
data/WineQT.csv
```

The dataset contains **1,143 wine samples** and physicochemical measurements of wine.
The original dataset contains the following columns:

- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality`
- `Id`

During training, the `Id` column is removed because it is only an identifier and does not provide useful predictive information.
Column names are normalized by replacing spaces with underscores.
For example, `fixed acidity` becomes `fixed_acidity`.

## Input features

The model uses **11 input features**:

| Feature | Description |
|---|---|
| `fixed_acidity` | Fixed acidity of the wine |
| `volatile_acidity` | Volatile acidity of the wine |
| `citric_acid` | Citric acid content |
| `residual_sugar` | Sugar remaining after fermentation |
| `chlorides` | Chloride content |
| `free_sulfur_dioxide` | Free sulfur dioxide amount |
| `total_sulfur_dioxide` | Total sulfur dioxide amount |
| `density` | Wine density |
| `pH` | Acidity/alkalinity level |
| `sulphates` | Sulphates content |
| `alcohol` | Alcohol percentage |

## Target classes

The original target column is `quality`, with numeric values from 3 to 8.
For this project, the original values are grouped into three classification classes:

| Original quality value | New class ID | New label |
|---|---:|---|
| 3, 4, 5 | 0 | `quality_low` |
| 6 | 1 | `quality_medium` |
| 7, 8 | 2 | `quality_high` |

This grouping is useful because the original dataset has rare extreme classes.
For example, very low and very high wine quality scores have fewer samples, so grouping them makes the classification task more stable.

## Model

The model is a scikit-learn `Pipeline`:

```text
StandardScaler → RandomForestClassifier
```

The classifier configuration:

```python
RandomForestClassifier(
    n_estimators=250,
    max_depth=8,
    random_state=42,
    class_weight="balanced",
)
```

Training uses an 80/20 stratified train-test split:

- 80% of the data is used for training.
- 20% of the data is used for testing.
- `random_state=42` is used for reproducibility.
- `stratify` is used to preserve class distribution in train and test sets.

## Model metrics

The current saved model has the following test metrics:

| Metric | Value |
|---|---:|
| Test accuracy | 0.681223 |
| Macro F1-score | 0.666853 |
| Weighted F1-score | 0.682608 |

Per-class scores:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `quality_low` | 0.7700 | 0.7333 | 0.7512 |
| `quality_medium` | 0.6082 | 0.6413 | 0.6243 |
| `quality_high` | 0.6250 | 0.6250 | 0.6250 |

The full classification report is saved in:

```text
artifacts/metrics.json
```

## 1. Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Train the model

Run:

```bash
python train.py
```

This script:

- loads `data/WineQT.csv`;
- removes the `Id` column;
- normalizes column names;
- creates a new target column called `quality_bin`;
- trains a `StandardScaler + RandomForestClassifier` pipeline;
- evaluates the model on the test set;
- saves model artifacts into the `artifacts/` directory;
- logs parameters, metrics, and the model to MLflow;
- registers the model in the MLflow Model Registry.

Generated artifacts:

```text
artifacts/model.joblib
artifacts/features.joblib
artifacts/target_names.joblib
artifacts/classes.joblib
artifacts/metrics.json
```

MLflow configuration used in `train.py`:

- Experiment name: `wine-quality`
- Registered model name: `wine-quality-classifier`

By default, MLflow writes runs to the local `./mlruns/` directory.

To log to a running MLflow tracking server, start MLflow first and set:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python train.py
```

On Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python train.py
```

## 3. Run the FastAPI service locally

Start the API:

```bash
uvicorn app.main:app --reload
```

Open in a browser:

- Root endpoint: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs

### `GET /`

Example response:

```json
{
  "message": "ML API is running"
}
```

### `GET /health`

Example response:

```json
{
  "status": "healthy",
  "model": "loaded"
}
```

### `POST /predict`

The prediction endpoint accepts 11 wine features and returns the predicted class, label, and class probabilities.

Request body:

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

Example response:

```json
{
  "predicted_class": 0,
  "predicted_label": "quality_low",
  "probabilities": {
    "quality_low": 0.810778,
    "quality_medium": 0.177041,
    "quality_high": 0.012181
  }
}
```

Test with curl:

```bash
curl http://127.0.0.1:8000/

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0.0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11.0,"total_sulfur_dioxide":34.0,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}'
```

## 4. Run the Streamlit frontend

Start the FastAPI backend first.

Then open a second terminal and run:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit app provides input fields for all 11 wine features, sends a request to `POST /predict`, and displays:

- predicted class label;
- numeric class ID;
- probability bar chart;
- probability values for each class.

The frontend reads the backend URL from the `API_URL` environment variable.
By default:

```text
http://localhost:8000
```

To use another backend URL:

```bash
API_URL=http://api:8000 streamlit run app.py
```

## 5. Run the MLflow UI

After running `python train.py`, you can open MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://localhost:5000
```

In MLflow you can inspect:

- experiment runs;
- logged hyperparameters;
- logged metrics;
- model artifacts;
- registered model versions.

Expected MLflow names:

| Item | Name |
|---|---|
| Experiment | `wine-quality` |
| Registered model | `wine-quality-classifier` |

## 6. Run with Docker: API only

Build the FastAPI image:

```bash
docker build -t practical-task-6 .
```

Run the container:

```bash
docker run -p 8000:8000 practical-task-6
```

Then open:

- http://localhost:8000/
- http://localhost:8000/health
- http://localhost:8000/docs

The Docker image includes the trained model artifacts, so it can serve predictions without retraining inside the container.

## 7. Run the full stack with Docker Compose

Run:

```bash
docker compose up --build
```

This starts three services:

| Service | URL | Description |
|---|---|---|
| MLflow | http://localhost:5000 | MLflow tracking server and Model Registry |
| API | http://localhost:8000 | FastAPI prediction service |
| Frontend | http://localhost:8501 | Streamlit user interface |

Inside the Docker Compose network, the Streamlit frontend uses:

```text
API_URL=http://api:8000
```

This works because `api` is the service name of the FastAPI container.

Stop the full stack:

```bash
docker compose down
```

## 8. Tests

Run smoke tests:

```bash
pytest -q
```

The tests check:

- `GET /`
- `GET /health`
- `POST /predict`

## Ports summary

| Port | Service |
|---:|---|
| 8000 | FastAPI API |
| 8501 | Streamlit UI |
| 5000 | MLflow UI |

## API validation

The request body is validated by Pydantic in `app/schemas.py`.
All fields are required.
Most numeric values must be greater than `0`, while `citric_acid` can be `0` or greater.

If the request body is invalid, FastAPI automatically returns a validation error.

## Short defense explanation

This project trains a machine learning model on the WineQT dataset and deploys it as a production-style ML service.
The training script prepares the data, converts the original wine quality scores into three classes, trains a scikit-learn pipeline, saves the model artifacts, and logs the experiment to MLflow.
The FastAPI backend loads the saved model and exposes prediction endpoints.
The Streamlit frontend allows users to interact with the model through a simple web interface.
Docker and Docker Compose make the project reproducible and easy to run with separate services for the API, frontend, and MLflow.
