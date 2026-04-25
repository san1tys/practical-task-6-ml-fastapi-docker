# Wine Classification — FastAPI, Docker, Streamlit, MLflow

This repository covers two connected assignments:

- **Practical Task 6** — train a machine learning model, save it, and deploy it as an API using FastAPI and Docker.
- **SIS-3 continuation** — extend the project with a Streamlit frontend and integrate MLflow (experiment tracking + Model Registry).

The model classifies wines into three classes using the `scikit-learn` Wine dataset.

## Technologies

- Python 3.11
- scikit-learn (RandomForestClassifier, StandardScaler)
- FastAPI + Uvicorn + Pydantic
- joblib
- Streamlit
- MLflow (tracking + Model Registry)
- Docker + Docker Compose

## Project structure

```text
practical-task-6-ml/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── model.py           # model loading + prediction
│   └── schemas.py         # Pydantic request/response models
├── artifacts/
│   ├── model.joblib
│   ├── features.joblib
│   ├── target_names.joblib
│   └── metrics.json
├── data/
│   └── wine_dataset.csv
├── frontend/
│   ├── app.py             # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── tests/
│   ├── conftest.py
│   └── test_smoke.py
├── mlruns/                # local MLflow store (created on first run)
├── train.py               # trains model + logs to MLflow
├── requirements.txt
├── Dockerfile             # FastAPI service image
├── docker-compose.yml     # MLflow + API + Streamlit
├── .dockerignore
└── README.md
```

## Dataset

The `scikit-learn` Wine dataset contains 178 samples with 13 chemical features (alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, proline). The target has three classes: `class_0`, `class_1`, `class_2`.

## Model

A single scikit-learn `Pipeline`:

```text
StandardScaler  →  RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42)
```

Training uses an 80/20 stratified split with `random_state=42` for full reproducibility.

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python train.py
```

This:

- loads the sklearn Wine dataset and writes it to `data/wine_dataset.csv`,
- trains the pipeline,
- saves `artifacts/model.joblib`, `artifacts/features.joblib`, `artifacts/target_names.joblib`, `artifacts/metrics.json`,
- creates an MLflow experiment named `wine-classification`,
- logs hyperparameters, metrics (`test_accuracy`, `macro_f1`, `weighted_f1`, plus per-class precision/recall/F1), and the model artifact,
- registers the model in the MLflow Model Registry as **`wine-classifier`**.

By default MLflow writes to a local `./mlruns/` directory (no server required). To target a remote tracking server, set the URI:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python train.py
```

## 3. Run the FastAPI service locally

```bash
uvicorn app.main:app --reload
```

Open in a browser:

- http://127.0.0.1:8000/ — root
- http://127.0.0.1:8000/health — health check
- http://127.0.0.1:8000/docs — Swagger UI

### `GET /`

```json
{ "message": "ML API is running" }
```

### `GET /health`

```json
{ "status": "healthy", "model": "loaded" }
```

### `POST /predict`

Request body (all 13 features required):

```json
{
  "alcohol": 14.23,
  "malic_acid": 1.71,
  "ash": 2.43,
  "alcalinity_of_ash": 15.6,
  "magnesium": 127.0,
  "total_phenols": 2.8,
  "flavanoids": 3.06,
  "nonflavanoid_phenols": 0.28,
  "proanthocyanins": 2.29,
  "color_intensity": 5.64,
  "hue": 1.04,
  "od280_od315_of_diluted_wines": 3.92,
  "proline": 1065.0
}
```

Example response:

```json
{
  "predicted_class": 0,
  "predicted_label": "class_0",
  "probabilities": {
    "class_0": 0.988,
    "class_1": 0.008,
    "class_2": 0.004
  }
}
```

Test from the command line:

```bash
curl http://127.0.0.1:8000/

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol":14.23,"malic_acid":1.71,"ash":2.43,"alcalinity_of_ash":15.6,"magnesium":127.0,"total_phenols":2.8,"flavanoids":3.06,"nonflavanoid_phenols":0.28,"proanthocyanins":2.29,"color_intensity":5.64,"hue":1.04,"od280_od315_of_diluted_wines":3.92,"proline":1065.0}'
```

## 4. Run the Streamlit frontend

In a second terminal (with the API already running):

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The app provides input fields for all 13 features, calls `POST /predict`, and shows the predicted class plus a probability bar chart.

The frontend reads its API URL from the `API_URL` environment variable (default `http://localhost:8000`). To point it elsewhere:

```bash
API_URL=http://api:8000 streamlit run app.py
```

## 5. Run the MLflow UI

After at least one `python train.py`, browse the experiment and Model Registry:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000:

- **Experiments → wine-classification** — runs with logged parameters, metrics, and the `model` artifact.
- **Models → wine-classifier** — registered model versions.

## 6. Run with Docker (API only)

```bash
docker build -t practical-task-6 .
docker run -p 8000:8000 practical-task-6
```

Then the same endpoints are reachable on the host:

- http://localhost:8000/
- http://localhost:8000/health
- http://localhost:8000/docs

The Docker image bundles the pre-trained `artifacts/model.joblib`, so no training step is required inside the container.

## 7. Run the full stack with Docker Compose

```bash
docker compose up --build
```

Three services come up:

| Service   | URL                    | Notes                                            |
|-----------|------------------------|--------------------------------------------------|
| MLflow    | http://localhost:5000  | Tracking server + Model Registry                 |
| API       | http://localhost:8000  | FastAPI service                                  |
| Frontend  | http://localhost:8501  | Streamlit UI (talks to API on the compose network) |

Inside the compose network the Streamlit container is configured with `API_URL=http://api:8000`, so it reaches the API by service name.

## 8. Tests

```bash
pytest -q
```

Three smoke tests cover `GET /`, `GET /health`, and `POST /predict`.

## Ports summary

| Port  | Service        |
|-------|----------------|
| 8000  | FastAPI API    |
| 8501  | Streamlit UI   |
| 5000  | MLflow UI      |

## Input features

All 13 features are required in `POST /predict` and are validated by Pydantic:

| Field                            | Description                          |
|----------------------------------|--------------------------------------|
| `alcohol`                        | Alcohol percentage                   |
| `malic_acid`                     | Malic acid (g/L)                     |
| `ash`                            | Ash content                          |
| `alcalinity_of_ash`              | Alcalinity of ash                    |
| `magnesium`                      | Magnesium (mg/L)                     |
| `total_phenols`                  | Total phenols                        |
| `flavanoids`                     | Flavanoid phenols                    |
| `nonflavanoid_phenols`           | Non-flavanoid phenols                |
| `proanthocyanins`                | Proanthocyanins                      |
| `color_intensity`                | Color intensity                      |
| `hue`                            | Hue                                  |
| `od280_od315_of_diluted_wines`   | OD280/OD315 absorbance ratio         |
| `proline`                        | Proline (mg/L)                       |
