from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Folders
DATA_DIR = BASE_DIR / "data"
DOC_DIR = BASE_DIR / "doc"
RESULTS_DIR = BASE_DIR / "results"

MODEL_OUTPUT_DIR = RESULTS_DIR / "model_outputs"

# Input files
FLIGHTS_PATH = DATA_DIR / "flights.csv"
AIRPORTS_PATH = DATA_DIR / "airports.csv"

# Output files
FINAL_DATASET_PATH = RESULTS_DIR / "final_dataset.csv"

# Model outputs
BASELINE_METRICS_PATH = MODEL_OUTPUT_DIR / "baseline_metrics.txt"
BASELINE_PLOT_PATH = MODEL_OUTPUT_DIR / "baseline_actual_vs_predicted.png"
BASELINE_MODEL_PATH = MODEL_OUTPUT_DIR / "baseline_model.joblib"

RF_METRICS_PATH = MODEL_OUTPUT_DIR / "rf_metrics.txt"
RF_PLOT_PATH = MODEL_OUTPUT_DIR / "rf_actual_vs_predicted.png"
RF_MODEL_PATH = MODEL_OUTPUT_DIR / "rf_model.joblib"

XGB_METRICS_PATH = MODEL_OUTPUT_DIR / "xgb_metrics.txt"
XGB_PLOT_PATH = MODEL_OUTPUT_DIR / "xgb_actual_vs_predicted.png"
XGB_MODEL_PATH = MODEL_OUTPUT_DIR / "xgb_model.joblib"

# NOAA settings
NOAA_STATION_ID = "GHCND:USW00024233"
NOAA_START_DATE = "2014-01-01"
NOAA_END_DATE = "2014-03-31"
NOAA_DATATYPES = ["TMAX", "PRCP"]