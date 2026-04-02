from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = RESULTS_DIR / "final_dataset.csv"
OUTPUT_DIR = RESULTS_DIR / "model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows")


# Features
print("Preparing features...")

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

X = df[feature_cols].copy()
y = df["arr_delay"].copy()


# Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model
print("\nTraining XGBoost... (this may take ~10-30s)")

xgb_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    ))
])

xgb_model.fit(X_train, y_train)

print("XGBoost training complete!")


# Predict
y_pred_xgb = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2 = r2_score(y_test, y_pred_xgb)


# Results
print("\n===== XGBOOST RESULTS =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")


# Save
text = f"""XGBoost Metrics

MAE  : {mae:.4f}
RMSE : {rmse:.4f}
R^2  : {r2:.4f}
"""
(OUTPUT_DIR / "xgb_metrics.txt").write_text(text)

# plot
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_xgb, alpha=0.25)

min_val = float(min(y_test.min(), y_pred_xgb.min()))
max_val = float(max(y_test.max(), y_pred_xgb.max()))
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Actual")
plt.ylabel("Predicted (XGB)")
plt.title("XGBoost: Actual vs Predicted")

plt.savefig(OUTPUT_DIR / "xgb_actual_vs_predicted.png", dpi=200)
plt.close()

print("Saved results!")