import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from main import train_and_forecast_arima


def test_train_and_forecast_arima_outputs_dataframe(tmp_path):
    # -------------------------
    # Arrange: create fake CSV
    # -------------------------
    dates = pd.date_range(start="2015-01-01", periods=60, freq="MS")
    inflation = np.random.normal(loc=2.0, scale=0.5, size=len(dates))

    df = pd.DataFrame({
        "date": dates,
        "inflation_rate": inflation
    })

    csv_path = tmp_path / "inflation.csv"
    df.to_csv(csv_path, index=False)

    # -------------------------
    # Act
    # -------------------------
    forecast_df = train_and_forecast_arima(
        csv_path=csv_path,
        start_forecast="2026-04-01",
        end_forecast="2026-12-01",
        order=(1, 1, 1)
    )

    # -------------------------
    # Assert
    # -------------------------
    # 1. Correct type
    assert isinstance(forecast_df, pd.DataFrame)

    # 2. Correct columns
    assert list(forecast_df.columns) == ["date", "predicted_inflation"]

    # 3. Dates are monthly and correct length
    expected_dates = pd.date_range(
        start="2026-04-01",
        end="2026-12-01",
        freq="MS"
    )
    assert len(forecast_df) == len(expected_dates)
    assert forecast_df["date"].equals(expected_dates)

    # 4. Forecast values exist and are numeric
    assert forecast_df["predicted_inflation"].notna().all()
    assert np.issubdtype(
        forecast_df["predicted_inflation"].dtype,
        np.number)
    
