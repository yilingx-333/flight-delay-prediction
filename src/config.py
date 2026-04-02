from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Folders
DATA_DIR = BASE_DIR / "data"
DOC_DIR = BASE_DIR / "doc"
RESULTS_DIR = BASE_DIR / "results"

# Input files
FLIGHTS_PATH = DATA_DIR / "flights.csv"
AIRPORTS_PATH = DATA_DIR / "airports.csv"

# Output files
FINAL_DATASET_PATH = RESULTS_DIR / "final_dataset.csv"

# NOAA settings
NOAA_STATION_ID = "GHCND:USW00024233"
NOAA_START_DATE = "2014-01-01"
NOAA_END_DATE = "2014-03-31"
NOAA_DATATYPES = ["TMAX", "PRCP"]