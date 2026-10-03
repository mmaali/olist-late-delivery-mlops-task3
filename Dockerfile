
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements/requirements-prod.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY config ./config
COPY artifacts/final_logistic_model_experiment2.joblib ./artifacts/final_logistic_model_experiment2.joblib
COPY artifacts/preprocessor_experiment2.joblib ./artifacts/preprocessor_experiment2.joblib
COPY artifacts/final_threshold_experiment2.json ./artifacts/final_threshold_experiment2.json
COPY artifacts/feature_list_experiment2.txt ./artifacts/feature_list_experiment2.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
