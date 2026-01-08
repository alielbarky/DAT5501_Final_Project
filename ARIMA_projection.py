import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA



def train_and_forecast_arima(
    csv_path,
    start_forecast="2026-04-01",
    end_forecast="2031-04-01",
    order=(1, 1, 1)
):
    # Load data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    inflation_series = df['inflation_rate']

    # Fit ARIMA on ALL available data
    model = ARIMA(inflation_series, order=order)
    fitted_model = model.fit()

    # Create forecast index (monthly)
    forecast_index = pd.date_range(
        start=start_forecast,
        end=end_forecast,
        freq='MS'
    )

    # Forecast
    forecast = fitted_model.forecast(steps=len(forecast_index))

    # Wrap into DataFrame
    forecast_df = pd.DataFrame({
        "date": forecast_index,
        "predicted_inflation": forecast.values
    })

    return forecast_df

forecast_df = train_and_forecast_arima(
    csv_path="data\processed\inflation.csv",
    order=(1, 1, 1)
)
forecast_df.to_csv("projected_inflation.csv", index=False)
print(forecast_df.head())
print(forecast_df.tail())