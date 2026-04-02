from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.load import load_api_key
from src.process import build_final_dataset, haversine


class TestProjectUtilities(unittest.TestCase):
    def test_load_api_key_from_env(self) -> None:
        """API key should be loaded from the NOAA_API_KEY environment variable."""
        with patch.dict(os.environ, {"NOAA_API_KEY": "test_key_123"}, clear=False):
            key = load_api_key()
            self.assertEqual(key, "test_key_123")


    def test_haversine_zero_distance(self) -> None:
        """Distance between identical coordinates should be zero."""
        lat = np.array([33.9416])
        lon = np.array([-118.4085])

        result = haversine(lat, lon, lat, lon)

        self.assertAlmostEqual(float(result[0]), 0.0, places=6)

    def test_haversine_known_distance(self) -> None:
        """A 1-degree longitude difference at the equator is about 111.2 km."""
        lat1 = np.array([0.0])
        lon1 = np.array([0.0])
        lat2 = np.array([0.0])
        lon2 = np.array([1.0])

        result = haversine(lat1, lon1, lat2, lon2)

        self.assertAlmostEqual(float(result[0]), 111.19, delta=1.0)

    def test_build_final_dataset_structure(self) -> None:
        """Final dataset builder should keep required columns and drop incomplete rows."""
        df = pd.DataFrame(
            {
                "month": [1, 1],
                "day": [1, 2],
                "origin": ["LAX", "SFO"],
                "dest": ["SFO", "LAX"],
                "carrier": ["AA", "DL"],
                "route": ["LAX_SFO", "SFO_LAX"],
                "dep_delay": [10, 5],
                "distance": [338, 338],
                "great_circle_km": [543.2, 543.2],
                "origin_lat": [33.9416, 37.6213],
                "origin_lon": [-118.4085, -122.3790],
                "dest_lat": [37.6213, 33.9416],
                "dest_lon": [-122.3790, -118.4085],
                "TMAX": [200, 210],
                "PRCP": [0, 5],
                "dep_hour": [8, 9],
                "is_weekend": [0, 1],
                "arr_delay": [12, 7],
            }
        )

        final_df = build_final_dataset(df)

        self.assertEqual(final_df.shape[0], 2)
        self.assertIn("arr_delay", final_df.columns)
        self.assertIn("route", final_df.columns)
        self.assertNotIn("year", final_df.columns)

    def test_prepared_dataset_if_exists(self) -> None:
        """
        If the final dataset exists, basic sanity checks should pass.
        This test does not fail if the file is absent, so it is safe during development.
        """
        dataset_path = Path("results/final_dataset.csv")
        if not dataset_path.exists():
            self.skipTest("results/final_dataset.csv not found")

        df = pd.read_csv(dataset_path)

        self.assertGreater(len(df), 0)
        self.assertIn("arr_delay", df.columns)
        self.assertIn("dep_delay", df.columns)
        self.assertIn("great_circle_km", df.columns)
        self.assertTrue(df["arr_delay"].notna().all())
        self.assertTrue(df["dep_delay"].notna().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)