"""
Скрипт тренування ML Pipeline для Bank Marketing Dataset.

Проводить 7 експериментів:
  - 3 експерименти XGBoost (різні гіперпараметри)
  - 2 експерименти Logistic Regression
  - 2 експерименти Random Forest
Всі результати логуються в MLflow.
Найкращу модель зберігає у models/best_pipeline.pkl.
"""
import os
import sys
import pickle
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

# Дозволяємо імпортувати pipeline з будь-якого місця
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.pipeline import (
    create_xgboost_pipeline,
    create_logistic_regression_pipeline,
    create_random_forest_pipeline,
)

DATASET_PATH = "data/raw/bank_marketing.csv"
MODELS_DIR = "models"
EXPERIMENT_NAME = "bank-marketing-classification"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_data(path: str = DATASET_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    print(f"Дані завантажено: {X.shape}, позитивних: {y.sum()} ({y.mean():.1%})")
    return X, y


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return metrics


def run_experiment(
    pipeline,
    X_train, y_train,
    X_test, y_test,
    params: dict,
    model_name: str,
    run_name: str,
) -> tuple[dict, str]:
    """Запустити один MLflow run та повернути метрики і run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "bank_marketing")

        mlflow.log_params({
            **params,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })

        # Cross-validation на тренувальній вибірці
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1
        )
        mlflow.log_metric("cv_f1_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_f1_std", float(cv_scores.std()))

        # Тренування
        pipeline.fit(X_train, y_train)

        # Метрики на train
        train_pred = pipeline.predict(X_train)
        train_metrics = compute_metrics(y_train, train_pred)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", float(v))

        # Метрики на test
        test_pred = pipeline.predict(X_test)
        test_prob = None
        try:
            test_prob = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            pass

        test_metrics = compute_metrics(y_test, test_pred, test_prob)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", float(v))

        print(f"\n{'='*55}")
        print(f"Run: {run_name}")
        print(f"  CV F1:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test Accuracy:{test_metrics['accuracy']:.4f}")
        print(f"  Test F1:      {test_metrics['f1']:.4f}")
        if "roc_auc" in test_metrics:
            print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(classification_report(y_test, test_pred, zero_division=0))

        # Логування моделі
        mlflow.sklearn.log_model(pipeline, "model")

        return {**test_metrics, "cv_f1_mean": cv_scores.mean()}, run.info.run_id


def save_best_model(pipeline, run_id: str, metrics: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_pipeline.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    meta = {"run_id": run_id, "metrics": {k: float(v) for k, v in metrics.items()}}
    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nНайкраща модель збережена: {model_path}")
    print(f"Run ID: {run_id}")


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    # ── XGBoost: 3 експерименти ─────────────────────────────────────────────
    xgb_configs = [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.9, "colsample_bytree": 0.7},
        {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08,
         "subsample": 0.85, "colsample_bytree": 0.85},
    ]
    for i, cfg in enumerate(xgb_configs, start=1):
        pipe = create_xgboost_pipeline(**cfg, random_state=RANDOM_STATE)
        metrics, run_id = run_experiment(
            pipe, X_train, y_train, X_test, y_test,
            params=cfg, model_name="XGBoost",
            run_name=f"xgboost_run_{i}",
        )
        results.append({"name": f"XGBoost run {i}", "params": cfg,
                        "metrics": metrics, "run_id": run_id, "pipeline": pipe})

    # ── Logistic Regression: 2 експерименти ────────────────────────────────
    lr_configs = [
        {"C": 0.1, "max_iter": 1000, "solver": "lbfgs"},
        {"C": 10.0, "max_iter": 2000, "solver": "lbfgs"},
    ]
    for i, cfg in enumerate(lr_configs, start=1):
        pipe = create_logistic_regression_pipeline(**cfg, random_state=RANDOM_STATE)
        metrics, run_id = run_experiment(
            pipe, X_train, y_train, X_test, y_test,
            params=cfg, model_name="LogisticRegression",
            run_name=f"logistic_regression_run_{i}",
        )
        results.append({"name": f"LogReg run {i}", "params": cfg,
                        "metrics": metrics, "run_id": run_id, "pipeline": pipe})

    # ── Random Forest: 2 експерименти ──────────────────────────────────────
    rf_configs = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
    ]
    for i, cfg in enumerate(rf_configs, start=1):
        pipe = create_random_forest_pipeline(**cfg, random_state=RANDOM_STATE)
        metrics, run_id = run_experiment(
            pipe, X_train, y_train, X_test, y_test,
            params=cfg, model_name="RandomForest",
            run_name=f"random_forest_run_{i}",
        )
        results.append({"name": f"Random Forest run {i}", "params": cfg,
                        "metrics": metrics, "run_id": run_id, "pipeline": pipe})

    # ── Вибір найкращої моделі ─────────────────────────────────────────────
    best = max(results, key=lambda r: r["metrics"].get("f1", 0))
    print(f"\n{'='*55}")
    print(f"НАЙКРАЩА МОДЕЛЬ: {best['name']}")
    print(f"  F1:      {best['metrics']['f1']:.4f}")
    print(f"  Accuracy:{best['metrics']['accuracy']:.4f}")
    if "roc_auc" in best["metrics"]:
        print(f"  ROC-AUC: {best['metrics']['roc_auc']:.4f}")
    print(f"  Run ID:  {best['run_id']}")

    save_best_model(best["pipeline"], best["run_id"], best["metrics"])

    # Зберегти порівняльну таблицю
    summary = []
    for r in results:
        row = {"run": r["name"], "run_id": r["run_id"], **r["params"]}
        row.update({f"test_{k}": v for k, v in r["metrics"].items()})
        summary.append(row)
    pd.DataFrame(summary).to_csv("models/experiments_summary.csv", index=False)
    print("\nЗведена таблиця збережена: models/experiments_summary.csv")

    return best


if __name__ == "__main__":
    main()
