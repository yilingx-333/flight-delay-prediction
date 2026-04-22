# AI generated:
import os
import time
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv


def load_api_key() -> str:
    """
    Load NOAA API key from the .env file only.
    """
    load_dotenv()

    key = os.getenv("NOAA_API_KEY")

    if not key:
        raise ValueError(
            "NOAA_API_KEY not found. Please set it in your .env file."
        )

    print("API key loaded from .env")
    return key.strip()


def load_flight_data(path: Path) -> pd.DataFrame:
    """
    Load and clean flight data.
    """
    flights = pd.read_csv(path)
    flights.columns = flights.columns.str.lower()

    print("Original shape:", flights.shape)

    flights = flights[
        (flights["year"] == 2014) &
        (flights["month"].isin([1, 2, 3]))
    ].copy()
    print("After filtering Jan–Mar:", flights.shape)

    flights = flights.dropna(
        subset=["arr_delay", "dep_time", "distance", "origin", "dest", "dep_delay"]
    )
    print("After dropping NA:", flights.shape)

    flights = flights[(flights["arr_delay"] < 300) & (flights["dep_delay"] < 300)].copy()
    print("After removing extreme delays:", flights.shape)

    return flights


def load_airport_data(path: Path) -> pd.DataFrame:
    """
    Load airport metadata and keep only IATA code + coordinates.
    """
    airports = pd.read_csv(path, header=None)
    airports.columns = [
        "AirportID", "Name", "City", "Country",
        "IATA", "ICAO", "Latitude", "Longitude",
        "Altitude", "Timezone", "DST",
        "Tz_database_time_zone", "Type", "Source"
    ]

    airports = airports[["IATA", "Latitude", "Longitude"]].copy()
    airports = airports.dropna(subset=["IATA", "Latitude", "Longitude"])

    print("Airport shape:", airports.shape)
    return airports


def fetch_noaa_weather(
    api_key: str,
    station_id: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch NOAA weather data with pagination.
    """
    print("Fetching NOAA weather data...")

    headers = {"token": api_key}
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": ["TMAX", "PRCP"],
        "limit": 1000,
        "offset": 1,
    }

    all_results = []

    while True:
        response = requests.get(base_url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Weather API error: {response.status_code} - {response.text}"
            )

        payload = response.json()
        results = payload.get("results", [])

        if not results:
            break

        all_results.extend(results)

        metadata = payload.get("metadata", {})
        resultset = metadata.get("resultset", {})
        count = resultset.get("count", 0)

        params["offset"] += len(results)

        if params["offset"] > count:
            break

        time.sleep(0.2)

    weather_df = pd.DataFrame(all_results)
    print("Weather raw shape:", weather_df.shape)
    return weather_df