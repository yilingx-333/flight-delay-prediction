from config import (
    AIRPORTS_PATH,
    FINAL_DATASET_PATH,
    FLIGHTS_PATH,
    NOAA_DATATYPES,
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


def main():
    print("========== DATA PREPARATION START ==========")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()

    flights = load_flight_data(FLIGHTS_PATH)
    airports = load_airport_data(AIRPORTS_PATH)

    flights = merge_airport_coords(flights, airports)
    flights = add_distance_features(flights)

    weather_df = fetch_noaa_weather(
        api_key=api_key,
        station_id=NOAA_STATION_ID,
        start_date=NOAA_START_DATE,
        end_date=NOAA_END_DATE,
    )

    weather_pivot = process_weather_data(weather_df)

    flights = merge_weather(flights, weather_pivot)
    flights = add_temporal_features(flights)

    final_df = build_final_dataset(flights)
    final_df.to_csv(FINAL_DATASET_PATH, index=False)

    print(f"Saved as {FINAL_DATASET_PATH}")
    print("========== DATA PREPARATION COMPLETE ==========")


if __name__ == "__main__":
    main()