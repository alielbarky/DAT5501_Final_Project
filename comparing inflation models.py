import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Splitting logic: Training <= 2013-01-01, Test > 2013-01-01 and <= 2019-01-01
    train = df[(df['date'] <= '2013-01-01')]
    test = df[(df['date'] > '2013-01-01') & (df['date'] <= '2019-01-01')]
    return train, test

def evaluate_best_polynomial(train, test):
    """Fits polynomials 1-9 and returns the best performing one."""
    X_train = np.arange(len(train))
    y_train = train['inflation_rate'].values
    X_test = np.arange(len(train), len(train) + len(test))
    y_test = test['inflation_rate'].values
    
    best_poly = {'mae': float('inf')}
    
    for order in range(1, 10):
        coeffs, cov = np.polyfit(X_train, y_train, deg=order, cov=True)
        y_pred = np.poly1d(coeffs)(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        if mae < best_poly['mae']:
            best_poly = {
                'order': order,
                'mae': mae,
                'coeffs': coeffs,
                'cov': cov
            }
    return best_poly

def arima_model_eval(train, test, order=(1,1,1)):
    """Fits ARIMA and returns the MAE."""
    model = ARIMA(train['inflation_rate'], order=order).fit()
    preds = model.forecast(steps=len(test))
    mae = mean_absolute_error(test['inflation_rate'], preds)
    return mae

# --- Execution ---
train, test = load_and_split_data("inflation.csv")

# 1. Evaluate Polynomials
best_poly = evaluate_best_polynomial(train, test)

# 2. Evaluate ARIMA
arima_mae = arima_model_eval(train, test)

# 3. Model Comparison Logic
if best_poly['mae'] < arima_mae:
    best_model_name = f"Polynomial (Order {best_poly['order']})"
    best_mae = best_poly['mae']
else:
    best_model_name = "ARIMA (1,1,1)"
    best_mae = arima_mae

# --- Reporting ---
print(f"--- Best Polynomial Model (Order {best_poly['order']}) ---")
print(f"MAE: {best_poly['mae']:.4f}")
print("Covariance Matrix of Coefficients:")
print(best_poly['cov'])

print(f"\n--- ARIMA Model ---")
print(f"MAE: {arima_mae:.4f}")

print("\n--- Final Comparison ---")
# The requested print statement:
print(f"the best overall model is {best_model_name} with a MAE of {best_mae:.4f}")
