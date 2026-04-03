from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_titanic_dataframe() -> pd.DataFrame:
    try:
        return sns.load_dataset("titanic")
    except Exception:
        local_csv = os.path.join(os.path.dirname(__file__), "data", "titanic.csv")
        return pd.read_csv(local_csv)


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df[["pclass", "sex", "age", "fare", "embarked"]].copy()
    y = df["survived"].astype(int).copy()
    return X, y


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    accuracy: float
    feature_importances: list[dict[str, Any]]


def train_model(random_state: int = 42) -> ModelBundle:
    df = load_titanic_dataframe()
    X, y = build_training_frame(df)

    numeric_features = ["age", "fare"]
    categorical_features = ["pclass", "sex", "embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=350,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    preprocess = pipeline.named_steps["preprocess"]
    rf = pipeline.named_steps["model"]

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(len(rf.feature_importances_))])

    importances = rf.feature_importances_
    pairs = list(zip(feature_names.tolist(), importances.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)

    top = pairs[:12]
    feature_importances = [{"feature": f, "importance": float(v)} for f, v in top]

    return ModelBundle(
        pipeline=pipeline,
        accuracy=acc,
        feature_importances=feature_importances,
    )


app = Flask(__name__)
_BUNDLE = train_model()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/model-info")
def model_info():
    return jsonify(
        {
            "accuracy": _BUNDLE.accuracy,
            "feature_importances": _BUNDLE.feature_importances,
        }
    )


def _parse_payload(payload: dict[str, Any]) -> pd.DataFrame:
    def as_float(x: Any) -> float | None:
        if x is None:
            return None
        if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)

    def as_int(x: Any) -> int | None:
        v = as_float(x)
        if v is None:
            return None
        return int(v)

    row = {
        "pclass": as_int(payload.get("pclass")),
        "sex": (payload.get("sex") or None),
        "age": as_float(payload.get("age")),
        "fare": as_float(payload.get("fare")),
        "embarked": (payload.get("embarked") or None),
    }
    return pd.DataFrame([row], columns=["pclass", "sex", "age", "fare", "embarked"])


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        X = _parse_payload(payload)
        proba = _BUNDLE.pipeline.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        confidence = float(np.max(proba))

        return jsonify(
            {
                "prediction": pred,
                "label": "Survived" if pred == 1 else "Did not survive",
                "confidence": confidence,
                "prob_survived": float(proba[1]),
                "prob_not_survived": float(proba[0]),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
