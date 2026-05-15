"""
Завантаження та підготовка Bank Marketing Dataset (UCI).

Датасет містить результати прямих маркетингових кампаній банку.
Ціль: передбачити, чи підпишеться клієнт на строковий депозит (y=1).
"""
import os
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np


OUTPUT_PATH = "data/raw/bank_marketing.csv"


def download_bank_marketing_dataset() -> pd.DataFrame:
    """
    Завантажити Bank Marketing Dataset з UCI Repository.
    Якщо завантаження не вдалося — генерує синтетичний датасет.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    print("Завантаження Bank Marketing Dataset з UCI...")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            csv_name = "bank-additional/bank-additional-full.csv"
            with z.open(csv_name) as f:
                df = pd.read_csv(f, sep=";")
        df = df.rename(columns={"y": "target"})
        df["target"] = (df["target"] == "yes").astype(int)
        print(f"Датасет завантажено: {df.shape[0]} рядків, {df.shape[1]} стовпців")
    except Exception as e:
        print(f"Не вдалося завантажити з UCI ({e}). Генерую синтетичний датасет...")
        df = _generate_synthetic_dataset()
    return df


def _generate_synthetic_dataset(n_samples: int = 4000, random_state: int = 42) -> pd.DataFrame:
    """Генерує синтетичний датасет, що імітує Bank Marketing."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 80, n_samples)
    duration = rng.integers(0, 2000, n_samples)
    campaign = rng.integers(1, 30, n_samples)
    pdays = rng.choice([999, *range(0, 30)], n_samples)
    previous = rng.integers(0, 10, n_samples)
    emp_var_rate = rng.uniform(-3.5, 1.5, n_samples)
    cons_price_idx = rng.uniform(92, 95, n_samples)
    cons_conf_idx = rng.uniform(-50, -26, n_samples)
    euribor3m = rng.uniform(0.6, 5.1, n_samples)
    nr_employed = rng.uniform(4963, 5228, n_samples)

    jobs = ["admin.", "blue-collar", "technician", "services", "management",
            "retired", "self-employed", "entrepreneur", "housemaid", "unemployed", "student"]
    marital = ["married", "single", "divorced"]
    education = ["basic.4y", "high.school", "basic.6y", "basic.9y",
                 "professional.course", "unknown", "university.degree", "illiterate"]
    default = ["no", "yes", "unknown"]
    housing = ["no", "yes", "unknown"]
    loan = ["no", "yes", "unknown"]
    contact = ["cellular", "telephone"]
    month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    day_of_week = ["mon", "tue", "wed", "thu", "fri"]
    poutcome = ["nonexistent", "failure", "success"]

    df = pd.DataFrame({
        "age": age,
        "job": rng.choice(jobs, n_samples),
        "marital": rng.choice(marital, n_samples),
        "education": rng.choice(education, n_samples),
        "default": rng.choice(default, n_samples),
        "housing": rng.choice(housing, n_samples),
        "loan": rng.choice(loan, n_samples),
        "contact": rng.choice(contact, n_samples),
        "month": rng.choice(month, n_samples),
        "day_of_week": rng.choice(day_of_week, n_samples),
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": rng.choice(poutcome, n_samples),
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    })

    # Генерація цільової змінної (дисбаланс класів ~11% позитивних)
    score = (
        0.003 * duration
        - 0.02 * campaign
        - 0.5 * emp_var_rate
        + 0.1 * (df["poutcome"] == "success").astype(int)
        + rng.normal(0, 1, n_samples)
    )
    threshold = np.percentile(score, 89)
    df["target"] = (score >= threshold).astype(int)

    print(f"Синтетичний датасет створено: {df.shape[0]} рядків, {df.shape[1]} стовпців")
    print(f"Розподіл класів: {df['target'].value_counts().to_dict()}")
    return df


def preprocess_and_save(df: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Датасет збережено: {path}")
    print(f"Форма: {df.shape}")
    print(f"Цільова змінна:\n{df['target'].value_counts()}")


if __name__ == "__main__":
    df = download_bank_marketing_dataset()
    preprocess_and_save(df)
