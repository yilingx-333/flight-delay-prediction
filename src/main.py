from config import (
    AIRPORTS_PATH,
    FINAL_DATASET_PATH,
    FLIGHTS_PATH,
    NOAA_END_DATE,
    NOAA_START_DATE,
    NOAA_STATION_ID,
    RESULTS_DIR,
)

from load import fetch_noaa_weather, load_airport_data, load_api_key, load_flight_data
from process import (
    add_distance_features,
    add_temporal_features,
    build_final_dataset,
    merge_airport_coords,
    merge_weather,
    process_weather_data,
)

# import modules
from analyze import main as run_analysis
from baseline import main as run_baseline
from random_forest import main as run_rf
from XGBoost import main as run_xgb


def run_pipeline() -> None:
    print("========== START FULL PIPELINE ==========")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # API
    api_key = load_api_key()

    # Load data
    flights = load_flight_data(FLIGHTS_PATH)
    airports = load_airport_data(AIRPORTS_PATH)

    # Feature engineering
    flights = merge_airport_coords(flights, airports)
    flights = add_distance_features(flights)

    # Weather
    weather_df = fetch_noaa_weather(
        api_key=api_key,
        station_id=NOAA_STATION_ID,
        start_date=NOAA_START_DATE,
        end_date=NOAA_END_DATE,
    )

    weather_pivot = process_weather_data(weather_df)

    flights = merge_weather(flights, weather_pivot)
    flights = add_temporal_features(flights)

    # Final dataset
    final_df = build_final_dataset(flights)
    final_df.to_csv(FINAL_DATASET_PATH, index=False)

    print("Dataset prepared!")

    # EDA
    print("\nRunning EDA...")
    run_analysis()

    # Models
    print("\nRunning baseline model...")
    run_baseline()

    print("\nRunning Random Forest...")
    run_rf()

    print("\nRunning XGBoost...")
    run_xgb()

    print("\n========== PIPELINE COMPLETE ==========")


def main():
    run_pipeline()

if __name__ == "__main__":
    main()