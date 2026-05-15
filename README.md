# MLOps Lab 1: Bank Marketing Classification

**Автор:** Student  
**Варіант:** 8  
**Дата:** 2026-05-15

## Опис проекту

Проект реалізує повний MLOps-цикл для задачі бінарної класифікації на основі **Bank Marketing Dataset (UCI)**. Мета — передбачити, чи підпишеться клієнт банку на строковий депозит (`target=1`).

Використовуються три типи моделей:
- **XGBoost Classifier** (основна, 3 експерименти)
- **Logistic Regression** (альтернативна, 2 експерименти)
- **Random Forest** (альтернативна, 2 експерименти)

Всі експерименти відстежуються через **MLflow**, дані та модель версіонуються через **DVC**.

---

## Структура проекту

```
lab1/
├── data/
│   ├── raw/                         # Оригінальні дані (під DVC)
│   │   ├── bank_marketing.csv.dvc   # DVC-метадані датасету
│   │   └── .gitkeep
│   ├── processed/                   # Оброблені дані
│   └── external/                    # Зовнішні дані
├── models/
│   ├── best_pipeline.pkl.dvc        # DVC-метадані найкращої моделі
│   ├── best_model_info.json         # Run ID та метрики найкращої моделі
│   └── experiments_summary.csv      # Зведена таблиця всіх експериментів
├── notebooks/                       # Jupyter notebooks
├── src/
│   ├── data/
│   │   └── create_dataset.py        # Завантаження Bank Marketing Dataset
│   ├── features/                    # Feature engineering (майбутнє)
│   └── models/
│       ├── pipeline.py              # Визначення sklearn Pipeline
│       ├── train.py                 # Скрипт тренування (7 експериментів)
│       └── utils.py                 # Утиліти завантаження/передбачення
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Датасет

**Bank Marketing Dataset (UCI)**  
- Джерело: https://archive.ics.uci.edu/ml/datasets/bank+marketing  
- Розмір: 41 188 рядків, 21 стовпець  
- Цільова змінна: `target` (0 — не підписав, 1 — підписав депозит)  
- Дисбаланс класів: ~11.3% позитивних

**Ознаки:**
- Числові: `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- Категоріальні: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`

---

## Швидкий старт

### Встановлення

```bash
git clone <repo-url>
cd lab1
python -m venv venv
source venv/Scripts/activate      # Windows
# або: source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### Отримання даних

```bash
dvc pull
```

### Запуск тренування

```bash
python src/data/create_dataset.py   # Завантажити датасет (якщо немає)
python src/models/train.py          # Запустити всі 7 експериментів
```

### Перегляд результатів в MLflow UI

```bash
mlflow ui --port 5000
# Відкрити http://localhost:5000
```

---

## Pipeline

Кожна модель обгортається в `sklearn.Pipeline`:

```
ColumnTransformer (preprocessor)
├── Числові ознаки → SimpleImputer(median) → StandardScaler
└── Категоріальні ознаки → SimpleImputer(most_frequent) → OneHotEncoder
        ↓
    Класифікатор (XGBoost / LogReg / RandomForest)
```

Перевага: **немає data leakage** — scaler навчається лише на тренувальних даних.

---

## Результати експериментів

| Run | Модель | Параметри | Test Accuracy | Test F1 | ROC-AUC | CV F1 |
|-----|--------|-----------|:-------------:|:-------:|:-------:|:-----:|
| 1 | **XGBoost** | n_est=100, depth=4, lr=0.1, sub=0.8 | **0.9239** | **0.9195** | **0.9551** | 0.9119 |
| 2 | XGBoost | n_est=200, depth=6, lr=0.05, sub=0.9 | 0.9223 | 0.9191 | 0.9554 | 0.9120 |
| 3 | XGBoost | n_est=150, depth=5, lr=0.08, sub=0.85 | 0.9222 | 0.9186 | 0.9557 | 0.9110 |
| 4 | Logistic Regression | C=0.1, max_iter=1000 | 0.9155 | 0.9059 | 0.9421 | 0.8998 |
| 5 | Logistic Regression | C=10.0, max_iter=2000 | 0.9164 | 0.9074 | 0.9423 | 0.9009 |
| 6 | Random Forest | n_est=100, depth=10 | 0.9121 | 0.8943 | 0.9442 | 0.8879 |
| 7 | Random Forest | n_est=200, depth=15 | 0.9173 | 0.9057 | 0.9503 | 0.8968 |

### Найкраща модель

- **Модель:** XGBoost Classifier (Run 1)
- **Параметри:** `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`
- **Test Accuracy:** 0.9239
- **Test F1 (weighted):** 0.9195
- **ROC-AUC:** 0.9551
- **CV F1 (5-fold):** 0.9119 ± 0.0024
- **Run ID:** `0c4a65782f5940ad9acf0e4c0e5ca53f`

---

## Використання моделі

```python
from src.models.utils import load_model_from_file, predict

# Завантажити найкращу модель
model = load_model_from_file('models/best_pipeline.pkl')

# Передбачення на нових даних (DataFrame з тими ж стовпцями)
predictions = predict(model, X_new)

# Або завантажити з MLflow
from src.models.utils import load_model_from_mlflow
model = load_model_from_mlflow('0c4a65782f5940ad9acf0e4c0e5ca53f')
```

---

## DVC Workflow

```bash
# Додати новий датасет
dvc add data/raw/bank_marketing.csv
git add data/raw/bank_marketing.csv.dvc
git commit -m "Update dataset"
dvc push

# Отримати дані на новій машині
git clone <repo-url>
dvc pull

# Перевірка статусу
dvc status
```

---

## Відтворюваність

Для забезпечення повної відтворюваності:
- Всі випадкові генератори використовують `random_state=42`
- Розбивка train/test: `test_size=0.2`, стратифікація за класом
- Версії всіх залежностей зафіксовані в `requirements.txt`
- Датасет і модель версіонуються через DVC

---

## Troubleshooting

| Проблема | Рішення |
|----------|---------|
| `dvc pull` не знаходить файли | Перевірте конфігурацію remote: `dvc remote list` |
| `ModuleNotFoundError: xgboost` | Активуйте venv та `pip install -r requirements.txt` |
| MLflow UI не стартує | Переконайтесь що порт 5000 вільний: `mlflow ui --port 5001` |
| Датасет не завантажується з UCI | Скрипт автоматично генерує синтетичний аналог |
| `dvc add` — bad DVC file name | Перевірте `.gitignore` — файл `.dvc` не повинен бути ігнорований |
