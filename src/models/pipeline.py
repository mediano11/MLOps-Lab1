"""
Модуль для створення ML Pipeline для Bank Marketing Dataset.

Включає preprocessing (OneHotEncoder для категорій, StandardScaler для числових)
та підтримку трьох класифікаторів: XGBoost, Logistic Regression, Random Forest.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost не встановлено, використовується GradientBoostingClassifier")
    from sklearn.ensemble import GradientBoostingClassifier

# Ознаки датасету Bank Marketing
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome"
]
NUMERICAL_FEATURES = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]


def _build_preprocessor() -> ColumnTransformer:
    """Побудувати трансформер для числових та категоріальних ознак."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def create_xgboost_pipeline(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """Створити Pipeline з XGBoost Classifier."""
    if XGBOOST_AVAILABLE:
        classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
    else:
        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
        )
    return Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier", classifier),
    ])


def create_logistic_regression_pipeline(
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    random_state: int = 42,
) -> Pipeline:
    """Створити Pipeline з Logistic Regression."""
    return Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )),
    ])


def create_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> Pipeline:
    """Створити Pipeline з Random Forest Classifier."""
    return Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )),
    ])


if __name__ == "__main__":
    pipe = create_xgboost_pipeline()
    print("XGBoost Pipeline steps:", list(pipe.named_steps.keys()))
    pipe_lr = create_logistic_regression_pipeline()
    print("Logistic Regression Pipeline steps:", list(pipe_lr.named_steps.keys()))
    pipe_rf = create_random_forest_pipeline()
    print("Random Forest Pipeline steps:", list(pipe_rf.named_steps.keys()))
