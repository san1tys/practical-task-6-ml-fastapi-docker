# Practical Task 6 — ML Model Deployment with FastAPI and Docker

This project was created for Practical Task 6.

The purpose of this task is to train a machine learning model, save it, and deploy it as an API using FastAPI and Docker.

In this project, I used the Wine dataset from `scikit-learn` and built a classification model that predicts the wine class based on input features.

## Project structure

```text
practical-task-6-better/
├── app/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── artifacts/
│   ├── features.joblib
│   ├── metrics.json
│   ├── model.joblib
│   └── target_names.joblib
├── data/
│   └── wine_dataset.csv
├── tests/
│   └── smoke_test.py
├── .dockerignore
├── Dockerfile
├── README.md
├── requirements.txt
└── train.py
```

## Description

This project includes:

- training a machine learning model,
- saving the trained model,
- creating a FastAPI application,
- making predictions through API endpoints,
- running the project in Docker.

## Dataset

For this practical task, the Wine dataset from `scikit-learn` was used.

It contains chemical features of wines, and the goal is to classify each sample into one of three classes.

## Model

The machine learning model is built using:

- `StandardScaler`
- `RandomForestClassifier`

These components are combined in a single `Pipeline`.

## Installation

First, create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

To train the model and save artifacts, run:

```bash
python train.py
```

After running this command, the project creates:

- trained model file,
- feature names file,
- target names file,
- evaluation metrics file,
- CSV version of the dataset.

## Run the FastAPI application

Start the API locally with:

```bash
uvicorn app.main:app --reload
```

After that, open these links in your browser:

- Root endpoint: `http://127.0.0.1:8000/`
- Health endpoint: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`

## API endpoints

### `GET /`

Returns a simple message showing that the API is running.

Example response:

```json
{
  "message": "ML API is running"
}
```

### `GET /health`

Returns the current status of the API.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts input features and returns the predicted wine class.

Example request:

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

## Testing

To run the included tests:

```bash
pytest -q
```

## Docker

### Build Docker image

```bash
docker build -t practical-task-6-better .
```

### Run Docker container

```bash
docker run -p 8000:8000 practical-task-6-better
```

After starting the container, open:

- `http://localhost:8000/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Conclusion

This project demonstrates a complete machine learning deployment workflow:

- training a model,
- saving the model,
- creating an API with FastAPI,
- testing the API locally,
- containerizing the application with Docker.
