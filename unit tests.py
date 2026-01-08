import unittest
import pandas as pd
import numpy as np
import tempfile
import os

from ARIMA_projection import train_and_forecast_arima


class TestTrainAndForecastARIMA(unittest.TestCase):

    def setUp(self):
        # Create a temporary CSV file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
        temp.close() # <-- CLOSE IT HERE to avoid Windows file-lock issues

        self.temp_file_name = temp.name

        # Generate fake monthly data
        dates = pd.date_range(start="2015-01-01", periods=60, freq="MS")
        inflation = np.random.normal(loc=2.0, scale=0.5, size=len(dates))

        df = pd.DataFrame({
            "date": dates,
            "inflation_rate": inflation
        })

        # Save to CSV
        df.to_csv(self.temp_file_name, index=False)

    def tearDown(self):
        # Delete temp file
        if os.path.exists(self.temp_file_name):
            os.unlink(self.temp_file_name)

    def test_forecast_output_structure(self):
        forecast_df = train_and_forecast_arima(
            csv_path=self.temp_file_name,
            start_forecast="2026-04-01",
            end_forecast="2026-12-01",
            order=(1, 1, 1)
        )

        # 1. Type check
        self.assertIsInstance(forecast_df, pd.DataFrame)

        # 2. Column check
        self.assertListEqual(
            list(forecast_df.columns),
            ["date", "predicted_inflation"])

        # 3. Length check
        expected_dates = pd.date_range(
            start="2026-04-01",
            end="2026-12-01",
            freq="MS"
        )
        self.assertEqual(len(forecast_df), len(expected_dates))

        # 4. Date correctness (ignore column name)
        pd.testing.assert_series_equal(
            forecast_df["date"],
            pd.Series(expected_dates),
            check_dtype=False,
            check_names=False)
        # 5. Forecast values sanity
        self.assertTrue(forecast_df["predicted_inflation"].notna().all())

if __name__ == "__main__":
    unittest.main()