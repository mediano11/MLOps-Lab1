"""
Допоміжні функції для роботи з моделями.
"""
import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn


def load_model_from_file(path: str):
    """Завантажити pipeline з .pkl файлу."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл моделі не знайдено: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Модель завантажено з: {path}")
    return model


def load_model_from_mlflow(run_id: str, artifact_path: str = "model"):
    """Завантажити модель з MLflow за run_id."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Модель завантажено з MLflow: {model_uri}")
    return model


def predict(model, X) -> np.ndarray:
    """Зробити передбачення. X може бути DataFrame або numpy array."""
    if isinstance(X, np.ndarray):
        return model.predict(X)
    return model.predict(X)


def predict_proba(model, X) -> np.ndarray:
    """Повернути ймовірності класів (якщо підтримується моделлю)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise AttributeError("Модель не підтримує predict_proba")


def get_best_run_id(experiment_name: str, metric: str = "test_f1") -> str:
    """Знайти run_id з найкращим значенням метрики в MLflow."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Експеримент '{experiment_name}' не знайдено")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("Runs не знайдено")
    return runs[0].info.run_id


if __name__ == "__main__":
    model_path = "models/best_pipeline.pkl"
    if os.path.exists(model_path):
        model = load_model_from_file(model_path)
        print(f"Pipeline steps: {list(model.named_steps.keys())}")
    else:
        print(f"Модель ще не натренована: {model_path}")
