from __future__ import annotations
from pathlib import Path
import argparse
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


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_INPUT_PATH = RESULTS_DIR / "final_dataset.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "model_outputs"


# Argument parsing
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a baseline flight delay model.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the cleaned dataset CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save model outputs.",
    )
    return parser.parse_args()


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
    y = df["arr_delay"].copy()
    X = df.drop(columns=["arr_delay"]).copy()

    # Drop constant columns if they exist.
    for col in ["year"]:
        if col in X.columns and X[col].nunique(dropna=False) <= 1:
            X = X.drop(columns=[col])

    # Separate feature types
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


# Output
def save_metrics(output_dir: Path, metrics: dict[str, float]) -> None:
    """Save evaluation metrics to a text file."""
    lines = [
        "Baseline Model Metrics",
        "=" * 22,
        f"MAE : {metrics['mae']:.4f}",
        f"RMSE: {metrics['rmse']:.4f}",
        f"R^2 : {metrics['r2']:.4f}",
        "",
        "Interpretation:",
        "- MAE is the average absolute prediction error in minutes.",
        "- RMSE penalizes larger errors more heavily.",
        "- R^2 measures how much variance is explained by the model.",
    ]
    (output_dir / "baseline_metrics.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_actual_vs_predicted(output_dir: Path, y_test: pd.Series, y_pred: np.ndarray) -> None:
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

    plt.savefig(output_dir / "baseline_actual_vs_predicted.png", dpi=200, bbox_inches="tight")
    plt.close()


# Main training routine
def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path).dropna(subset=["arr_delay"]).copy()

    X, y, numeric_cols, categorical_cols = build_feature_sets(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Console output
    print("===== BASELINE MODEL RESULTS =====")
    print(f"Rows used: {len(df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Train size: {len(X_train):,}")
    print(f"Test size : {len(X_test):,}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    # Save outputs
    save_metrics(output_dir, {"mae": mae, "rmse": rmse, "r2": r2})
    plot_actual_vs_predicted(output_dir, y_test, y_pred)

    print(f"Saved metrics to: {output_dir / 'baseline_metrics.txt'}")
    print(f"Saved plot to: {output_dir / 'baseline_actual_vs_predicted.png'}")


if __name__ == "__main__":
    main()