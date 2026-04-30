import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for ML modeling.
    Only resolved tickets and no null lead time.
    """

    df = df[df["status"].str.lower() == "release ready"].copy()
    df = df.dropna(subset=["lead_time_days"])

    # Temporal feature engineering
    df["created_month"] = df["created"].dt.month
    df["created_weekday"] = df["created"].dt.weekday
    df["created_quarter"] = df["created"].dt.quarter

    return df


def train_lead_time_model(df: pd.DataFrame):
    """
    Train RandomForest model to predict lead_time_days.
    Returns trained pipeline and evaluation metrics.
    """

    df = prepare_ml_dataset(df)

    target = "lead_time_days"

    features = [
        "priority",
        "created_month",
        "created_weekday",
        "created_quarter"
    ]

    X = df[features]
    y = df[target]

    categorical_features = ["priority"]
    numeric_features = ["created_month", "created_weekday", "created_quarter"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "R2": round(r2_score(y_test, y_pred), 3),
        "Baseline_MAE": round(np.abs(y_test - y_test.mean()).mean(), 2)
    }

    return pipeline, metrics, y_test, y_pred
