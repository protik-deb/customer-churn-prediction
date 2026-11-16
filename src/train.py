"""Train baseline customer churn classifiers.

Example:
    python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Churn"
DROP_COLUMNS = ["customerID"]


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and validate the target column."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See data/README.md for instructions."
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column {TARGET!r}; found {list(df.columns)}")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted only on each training fold."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def evaluate(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Print and return a concise set of test metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print(f"\n{'=' * 16} {name} {'=' * 16}")
    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric:>10}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to the Telco Customer Churn CSV.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data).copy()
    df = df.drop_duplicates()

    # TotalCharges is numeric in meaning but may contain blank strings.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn target column.")

    X = df.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=y,
    )

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=args.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_name = str(results_df.iloc[0]["model"])
    best_model = fitted_models[best_name]
    output_path = Path("models") / "best_model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f"\nSaved best model ({best_name}) to {output_path}")


if __name__ == "__main__":
    main()
