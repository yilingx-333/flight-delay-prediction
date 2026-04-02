from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = RESULTS_DIR / "final_dataset.csv"
OUTPUT_DIR = RESULTS_DIR / "model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load data
def load_data(path: Path) -> pd.DataFrame:
    """Load the cleaned dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if "arr_delay" not in df.columns:
        raise ValueError("Target column 'arr_delay' not found.")

    return df


# Prepare features
def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    feature_cols = [
        "dep_delay",
        "distance",
        "great_circle_km",
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
        "TMAX",
        "PRCP",
        "dep_hour",
        "is_weekend",
        "month",
        "day",
    ]

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X = df[feature_cols].copy()
    y = df["arr_delay"].copy()

    return X, y


# Save outputs
def save_metrics(output_dir: Path, mae: float, rmse: float, r2: float) -> None:
    """Save model metrics to a text file."""
    text = f"""Random Forest Metrics

MAE  : {mae:.4f}
RMSE : {rmse:.4f}
R^2  : {r2:.4f}
"""
    (output_dir / "rf_metrics.txt").write_text(text, encoding="utf-8")


def save_plot(output_dir: Path, y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Save actual vs predicted scatter plot."""
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.25)

    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual Arrival Delay")
    plt.ylabel("Predicted Arrival Delay")
    plt.title("Random Forest: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "rf_actual_vs_predicted.png", dpi=200, bbox_inches="tight")
    plt.close()


# Main
def main() -> None:
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("Preparing features...")
    X, y = build_feature_matrix(df)
    print(f"Using {X.shape[1]} numeric features")

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    print(f"Train size: {len(X_train):,}")
    print(f"Test size : {len(X_test):,}")

    print("Training Random Forest... (this may take a moment)")
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=20,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    print("Training complete.")

    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== RANDOM FOREST RESULTS =====")
    print(f"Rows used: {len(df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    save_metrics(OUTPUT_DIR, mae, rmse, r2)
    save_plot(OUTPUT_DIR, y_test, y_pred)

    print(f"Saved metrics to: {OUTPUT_DIR / 'rf_metrics.txt'}")
    print(f"Saved plot to: {OUTPUT_DIR / 'rf_actual_vs_predicted.png'}")


if __name__ == "__main__":
    main()