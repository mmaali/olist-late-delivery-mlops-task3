# Olist Late Delivery Prediction - Task 3

This repository contains the standalone Task 3 inference service for the Olist late-delivery model. It includes the API, MLflow setup, Docker Compose configuration, runtime model artifacts, tests, and all nine Olist CSV datasets.

## Quick start

With Docker Desktop running, run:

```powershell
docker compose up -d --build
docker compose ps -a
```

- FastAPI: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

Check API health at http://localhost:8000/health. See [the full Task 3 report](docs/Task3_Report.md) for the architecture, configuration, validation evidence, API examples, monitoring, CI/CD, and project structure. The PDF version is [docs/Task3_Report.pdf](docs/Task3_Report.pdf).

The API uses `olist db/olist_geolocation_dataset.csv`; the complete dataset is also available under `data/`.