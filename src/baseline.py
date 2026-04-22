# AI generated:
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Support both direct script execution and imports
try:
    from config import FINAL_DATASET_PATH, RESULTS_DIR
except ImportError:
    from src.config import FINAL_DATASET_PATH, RESULTS_DIR


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_OUTPUT_DIR = RESULTS_DIR / "model_outputs"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_METRICS_PATH = MODEL_OUTPUT_DIR / "baseline_metrics.txt"
BASELINE_PLOT_PATH = MODEL_OUTPUT_DIR / "baseline_actual_vs_predicted.png"


# Data loading
def load_data(path: Path) -> pd.DataFrame:
    """Load the cleaned dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    if "arr_delay" not in df.columns:
        raise ValueError("Target column 'arr_delay' not found in dataset.")

    return df


# Feature preparation
def build_feature_sets(df: pd.DataFrame):
    """Split target and features, and identify numeric/categorical columns."""
    y = df["arr_delay"].copy()
    X = df.drop(columns=["arr_delay"]).copy()

    # Drop constant columns if they exist.
    for col in ["year"]:
        if col in X.columns and X[col].nunique(dropna=False) <= 1:
            X = X.drop(columns=[col])

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    return X, y, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical columns."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


# Output helpers
def save_metrics(mae: float, rmse: float, r2: float) -> None:
    """Save evaluation metrics to a text file."""
    lines = [
        "Baseline Model Metrics",
        "=" * 22,
        f"MAE : {mae:.4f}",
        f"RMSE: {rmse:.4f}",
        f"R^2 : {r2:.4f}",
        "",
        "Interpretation:",
        "- MAE is the average absolute prediction error in minutes.",
        "- RMSE penalizes larger errors more heavily.",
        "- R^2 measures how much variance is explained by the model.",
    ]
    BASELINE_METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Save a scatter plot of actual vs predicted values."""
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.25)

    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual Arrival Delay")
    plt.ylabel("Predicted Arrival Delay")
    plt.title("Baseline Actual vs Predicted Arrival Delay")
    plt.tight_layout()

    plt.savefig(BASELINE_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()


# Main training routine
def main() -> None:
    print("Loading dataset...")
    df = load_data(FINAL_DATASET_PATH).dropna(subset=["arr_delay"]).copy()
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("Preparing features...")
    X, y, numeric_cols, categorical_cols = build_feature_sets(df)
    print(f"Using {X.shape[1]} features")

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    print(f"Train size: {len(X_train):,}")
    print(f"Test size : {len(X_test):,}")

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    print("Training baseline model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Training complete.")

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== BASELINE MODEL RESULTS =====")
    print(f"Rows used: {len(df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    # Save outputs
    save_metrics(mae, rmse, r2)
    plot_actual_vs_predicted(y_test, y_pred)

    print(f"Saved metrics to: {BASELINE_METRICS_PATH}")
    print(f"Saved plot to: {BASELINE_PLOT_PATH}")


if __name__ == "__main__":
    main()