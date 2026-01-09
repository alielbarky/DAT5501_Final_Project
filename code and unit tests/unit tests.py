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
        temp.close() # close it again to avoid Windows file-lock

        self.temp_file_name = temp.name

        # Generate simulated monthly data for testing
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

        #check type
        self.assertIsInstance(forecast_df, pd.DataFrame)

        #check column
        self.assertListEqual(
            list(forecast_df.columns),
            ["date", "predicted_inflation"])

        #check length
        expected_dates = pd.date_range(
            start="2026-04-01",
            end="2026-12-01",
            freq="MS"
        )
        self.assertEqual(len(forecast_df), len(expected_dates))

        #check date
        pd.testing.assert_series_equal(
            forecast_df["date"],
            pd.Series(expected_dates),
            check_dtype=False,
            check_names=False)
        
        self.assertTrue(forecast_df["predicted_inflation"].notna().all())

if __name__ == "__main__":
    unittest.main()