# MLOps Lab 1: Bank Marketing Classification

**Автор:** Пестенков Дмітрій Олександрович  
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
│   ├── raw/                         # Оригінальні дані (CSV під DVC, у Git лише *.dvc)
│   │   ├── bank_marketing.csv.dvc
│   │   └── .gitkeep
│   ├── processed/
│   └── external/
├── models/
│   ├── best_pipeline.pkl.dvc        # Найкращий пайплайн (під DVC)
│   ├── best_model_info.json         # MLflow Run ID + метрики
│   └── experiments_summary.csv      # Експортується після train.py
├── scripts/
│   └── run_mlflow_ui.py             # MLflow UI на Windows з коректними MIME для .js
├── notebooks/
├── src/
│   ├── data/
│   │   └── create_dataset.py        # Завантаження датасету з UCI (або синтетика)
│   ├── features/
│   └── models/
│       ├── pipeline.py
│       ├── train.py                 # 7 експериментів + збереження best_pipeline.pkl
│       └── utils.py
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

## Інструкція для нових користувачів

Нижче — повний порядок дій із нуля до відтворених результатів.

### 1. Клонування репозиторію

```bash
git clone git@github.com:mediano11/MLOps-Lab1.git
cd lab1
```

Переконайтеся, що ви в **корені проєкту** (`lab1`), де лежать `requirements.txt`, `src/`, `data/`.

### 2. Python і віртуальне середовище

Потрібен **Python 3.12** (або сумісний 3.x з `requirements.txt`).

```bash
python -m venv venv
```

**Активація venv:**

| Оболонка        | Команда                        |
| --------------- | ------------------------------ |
| Git Bash / MSYS | `source venv/Scripts/activate` |
| CMD             | `venv\Scripts\activate.bat`    |
| PowerShell      | `venv\Scripts\Activate.ps1`    |
| Linux / macOS   | `source venv/bin/activate`     |

### 3. Встановлення залежностей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Перевірка:

```bash
python -c "import mlflow, dvc; import sklearn; print('OK')"
```

### 4. Отримання даних через DVC

Якщо у репозиторії налаштований **віддалений DVC-remote** і у вас є доступ:

```bash
dvc pull
```

Після успішного `dvc pull` з’явиться файл `data/raw/bank_marketing.csv` (він у `.gitignore`, у Git лише `*.dvc`).

**Якщо `dvc pull` недоступний** (нема доступу до сховища автора або remote не налаштований):

```bash
python src/data/create_dataset.py
```

Це створить `data/raw/bank_marketing.csv` локально. Далі можете додати його під власний DVC (`dvc remote add …`, `dvc add …`, `dvc push`), якщо потрібно повторити схему версіонування.

### 5. Запуск тренування

Перебуваючи в **корені проєкту** з активованим venv:

```bash
python src/models/train.py
```

Очікуваний результат:

- у каталозі `mlruns/` з’являться runs експерименту **bank-marketing-classification**;
- збережеться `models/best_pipeline.pkl`, `models/best_model_info.json`, `models/experiments_summary.csv`.

**Примітка.** Якщо змінити код або дані й знову зберегти найкращу модель під DVC, виконуйте звичний робочий цикл з `steps.md`: `dvc add models/best_pipeline.pkl`, `git add` файл `*.dvc`, `git commit`, `dvc push`.

### 6. Перегляд експериментів у MLflow UI

У Windows браузер може блокувати інтерфейс через некоректний MIME для `.js`. Рекомендовано:

```bash
python scripts/run_mlflow_ui.py
```

Потім у браузері: **http://127.0.0.1:5000/** .

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

Перевага: **немає data leakage** — трансформації підлаштовуються лише на тренувальній частині.

---

## Результати експериментів

Повну таблицю після локального запуску дивіться також у `models/experiments_summary.csv`.

| Run | Модель              | Параметри                                                 | Test Accuracy |  Test F1   |  ROC-AUC   | CV F1 (mean, 5-fold) |
| --- | ------------------- | --------------------------------------------------------- | :-----------: | :--------: | :--------: | :------------------: |
| 1   | **XGBoost**         | n_est=100, max_depth=4, lr=0.1, sub=0.8, colsample=0.8    |  **0.9239**   | **0.9195** | **0.9551** |        0.9119        |
| 2   | XGBoost             | n_est=200, max_depth=6, lr=0.05, sub=0.9, colsample=0.7   |    0.9223     |   0.9191   |   0.9554   |        0.9120        |
| 3   | XGBoost             | n_est=150, max_depth=5, lr=0.08, sub=0.85, colsample=0.85 |    0.9222     |   0.9186   |   0.9557   |        0.9110        |
| 4   | Logistic Regression | C=0.1, max_iter=1000, solver=lbfgs                        |    0.9155     |   0.9059   |   0.9421   |        0.8998        |
| 5   | Logistic Regression | C=10.0, max_iter=2000, solver=lbfgs                       |    0.9164     |   0.9074   |   0.9423   |        0.9009        |
| 6   | Random Forest       | n_est=100, max_depth=10, min_samples_split=2              |    0.9121     |   0.8943   |   0.9442   |        0.8879        |
| 7   | Random Forest       | n_est=200, max_depth=15, min_samples_split=5              |    0.9173     |   0.9057   |   0.9503   |        0.8968        |

### Найкраща модель (документовані параметри й метрики)

Обрано за максимумом **test F1 (weighted)** на утриманій тестовій вибірці після навчання.

| Поле                   | Значення                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| **Тип моделі**         | XGBoost Classifier (разом із препроцесингом у одному sklearn `Pipeline`) |
| **Експеримент MLflow** | `bank-marketing-classification`                                          |
| **Ім’я run**           | `xgboost_run_1`                                                          |
| **Run ID**             | `0c4a65782f5940ad9acf0e4c0e5ca53f`                                       |

**Гіперпараметри класифікатора (ядро XGBoost):**

| Параметр           | Значення |
| ------------------ | -------- |
| `n_estimators`     | 100      |
| `max_depth`        | 4        |
| `learning_rate`    | 0.1      |
| `subsample`        | 0.8      |
| `colsample_bytree` | 0.8      |
| `random_state`     | 42       |

**Фіксація експерименту й даних:**

| Параметр                             | Значення                                |
| ------------------------------------ | --------------------------------------- |
| `train_test_split.test_size`         | 0.2                                     |
| `train_test_split.random_state`      | 42                                      |
| Стратифікація за `target`            | так                                     |
| Cross-validation (`cross_val_score`) | 5 folds, метрика `f1_weighted` на train |
| Шлях до даних                        | `data/raw/bank_marketing.csv`           |

**Метрики на тесті (останній успішний прогін; дублікат у `models/best_model_info.json`):**

| Метрика              | Значення   |
| -------------------- | ---------- |
| Accuracy             | 0.9239     |
| Precision (weighted) | 0.9178     |
| Recall (weighted)    | 0.9239     |
| F1 (weighted)        | **0.9195** |
| ROC-AUC              | **0.9551** |

**Середнє CV F1 (weighted) на train:** 0.9120

**Файли артефактів:**

- Локально після навчання: `models/best_pipeline.pkl`
- Версіонування даних указником DVC: `models/best_pipeline.pkl.dvc` (+ `dvc pull` відновлює бінарник)

---

## Використання моделі

З кореня проєкту й активним venv:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from models.utils import load_model_from_file, predict

model = load_model_from_file("models/best_pipeline.pkl")
predictions = predict(model, X_new)
```

Через MLflow:

```python
from models.utils import load_model_from_mlflow

model = load_model_from_mlflow("0c4a65782f5940ad9acf0e4c0e5ca53f")
```

---

## DVC Workflow (додатково)

```bash
# Оновити датасет і відправити в remote
dvc add data/raw/bank_marketing.csv
git add data/raw/bank_marketing.csv.dvc
git commit -m "Update dataset"
dvc push

# Нова машина після clone
git clone git@github.com:mediano11/MLOps-Lab1.git && cd lab1
pip install -r requirements.txt   # і venv за потреби
dvc pull
```

Корисні команди: `dvc status`, `dvc remote list`, `dvc doctor`.

---

## Відтворюваність

- `random_state=42` у розбитті даних і де це підтримує модель
- Залежності зафіксовані в `requirements.txt`
- Версії даних і моделі — через DVC та MLflow (`run_id`, параметри й метрики в UI)

---

## Troubleshooting

| Проблема                                                                         | Ймовірна причина                                                           | Що зробити                                                                                                                                                     |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Порожній / білий MLflow UI, у консолі браузера: MIME type `text/plain` для `.js` | На Windows некоректне визначення MIME для статичних файлів Python-сервером | Запускати `python scripts/run_mlflow_ui.py`, відкрити **http://127.0.0.1:5000/** ; не використовувати `view-source`; спробувати інкогніто без крипто-розширень |
| `dvc pull` не завантажує файли                                                   | Немає доступу до remote або remote не налаштований після fork              | `dvc remote list`; якщо треба — `dvc remote add -d <name> <url-або-шлях>`; або згенерувати дані: `python src/data/create_dataset.py`                           |
| `ERROR: bad DVC file name ... is git-ignored`                                    | У `.gitignore` ігноруються `*.dvc` у потрібній теці                        | Дозволити винятки для `data/raw/*.dvc`, `models/*.dvc` (як у цьому репозиторії)                                                                                |
| `FileNotFoundError: ... bank_marketing.csv`                                      | Дані ще не завантажені                                                     | `dvc pull` або `python src/data/create_dataset.py`                                                                                                             |
| `ModuleNotFoundError` (mlflow, xgboost, sklearn)                                 | Не активовано venv або не встановлено залежності                           | `source venv/Scripts/activate` (або аналог), потім `pip install -r requirements.txt`                                                                           |
| Помилки імпорту при запуску `train.py` з іншої теки                              | Скрипт очікує робочу директорію кореня проєкту                             | `cd lab1` і запуск `python src/models/train.py`                                                                                                                |
| Порт 5000 зайнятий                                                               | Інший процес слухає порт                                                   | `python scripts/run_mlflow_ui.py ui --host 127.0.0.1 --port 5001` або змінити порт у скрипті                                                                   |
| Не вдається завантажити CSV з UCI                                                | Мережа / файрвол                                                           | Скрипт `create_dataset.py` має fallback на синтетичні дані                                                                                                     |
| Відрізняються метрики після повторного запуску                                   | Змінені дані, версія бібліотек або не той `random_state`                   | Оновити середовище з `requirements.txt`, переконатися в тому ж `bank_marketing.csv` і коді `train.py`                                                          |
