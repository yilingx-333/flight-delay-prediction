# AI generated:
import numpy as np
import pandas as pd


def merge_airport_coords(flights: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    origin_coords = airports.rename(columns={
        "IATA": "origin",
        "Latitude": "origin_lat",
        "Longitude": "origin_lon"
    })

    dest_coords = airports.rename(columns={
        "IATA": "dest",
        "Latitude": "dest_lat",
        "Longitude": "dest_lon"
    })

    flights = flights.merge(origin_coords, on="origin", how="left")
    flights = flights.merge(dest_coords, on="dest", how="left")

    print("After airport merge:", flights.shape)
    return flights


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def add_distance_features(flights: pd.DataFrame) -> pd.DataFrame:
    print("Calculating great-circle distance...")

    flights = flights.dropna(subset=["origin_lat", "origin_lon", "dest_lat", "dest_lon"]).copy()
    flights["great_circle_km"] = haversine(
        flights["origin_lat"],
        flights["origin_lon"],
        flights["dest_lat"],
        flights["dest_lon"]
    )

    return flights


def process_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        raise ValueError("No weather data returned from NOAA API.")

    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

    weather_pivot = weather_df.pivot_table(
        index="date",
        columns="datatype",
        values="value",
        aggfunc="mean"
    ).reset_index()

    weather_pivot.columns.name = None

    print("Weather processed shape:", weather_pivot.shape)
    return weather_pivot


def merge_weather(flights: pd.DataFrame, weather_pivot: pd.DataFrame) -> pd.DataFrame:
    flights = flights.copy()
    flights["flight_date"] = pd.to_datetime(flights[["year", "month", "day"]]).dt.date

    flights = flights.merge(
        weather_pivot,
        left_on="flight_date",
        right_on="date",
        how="left"
    )

    print("After weather merge:", flights.shape)
    return flights


def add_temporal_features(flights: pd.DataFrame) -> pd.DataFrame:
    print("Creating features...")

    flights = flights.copy()
    flights["dep_hour"] = (pd.to_numeric(flights["dep_time"], errors="coerce") // 100).astype("Int64")
    flights["weekday"] = pd.to_datetime(flights[["year", "month", "day"]]).dt.weekday
    flights["is_weekend"] = flights["weekday"].isin([5, 6]).astype(int)
    flights["route"] = flights["origin"].astype(str) + "_" + flights["dest"].astype(str)

    return flights


def build_final_dataset(flights: pd.DataFrame) -> pd.DataFrame:
    final_columns = [
        "month", "day",
        "origin", "dest", "carrier", "route",
        "dep_delay",
        "distance",
        "great_circle_km",
        "origin_lat", "origin_lon",
        "dest_lat", "dest_lon",
        "TMAX", "PRCP",
        "dep_hour", "is_weekend",
        "arr_delay"
    ]

    final_columns = [c for c in final_columns if c in flights.columns]
    final_df = flights[final_columns].dropna().copy()

    print("Final dataset shape:", final_df.shape)
    return final_df