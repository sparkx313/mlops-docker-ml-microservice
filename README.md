# Dockerized ML Model Service

A small HTTP service that loads a fitted binary classifier and serves predictions through FastAPI. Train the model from labeled data you supply; no model or generated dataset is bundled.

## 1. Train a model

Requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
python train_model.py --data path/to/labeled_data.csv --target label --id-column record_id
```

The target column must contain numeric `0` and `1` labels. Other columns become model features, except the optional ID column. The trainer creates a stratified holdout, fits a preprocessing and logistic-regression pipeline, reports ROC AUC, average precision, and Brier score, and saves `model.joblib` plus holdout outputs. Use `--model`, `--metrics`, `--predictions`, `--test-size`, and `--seed` to configure those outputs and the split.

## 2. Run the API locally

The current directory must contain the model artifact, or set `MODEL_PATH` to its location:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

The API provides:

- `GET /health` reports whether the model loaded.
- `GET /metadata` reports the target, feature schema, and holdout metrics.
- `POST /predict` accepts a JSON object with a `features` map matching the model schema, then returns the predicted label and class probabilities.
- `GET /docs` provides the FastAPI request schema.

## Docker

Build the multi-stage image:

```bash
docker build -t ml-model-service .
```

Mount the trained artifact read-only and start the service:

```bash
docker run --rm -p 8000:8000 -v /absolute/path/to/model-directory:/models:ro -e MODEL_PATH=/models/model.joblib ml-model-service
```

The image health check calls `/health`. The container exits during startup if the configured model artifact is missing or has an unsupported format.

## Scope and operations

This is an unauthenticated binary-classification API intended to be run on a trusted network. Add authentication and network controls before exposing it to untrusted clients. Prediction probabilities and holdout metrics depend on the supplied data; this repository does not claim calibrated probabilities, production scale, or suitability for consequential decisions.

Only load model artifacts you trust. The API reads the model locally and makes no outbound network requests.

## License

MIT. See [LICENSE](LICENSE).
