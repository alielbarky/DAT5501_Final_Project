import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Define a function to load and define data
def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Split the data to train and test
    train = df[(df['date'] <= '2013-01-01')]
    test = df[(df['date'] > '2013-01-01') & (df['date'] <= '2019-01-01')]
    return train, test

def compute_metrics(y_true, y_pred, k):
    n = len(y_true)
    rss = np.sum((y_true - y_pred)**2)
    dof = n - k
    # Chi-squared per degree of freedom (assuming sigma=1)
    chi2_dof = rss / dof if dof > 0 else np.inf
    # BIC = n * ln(RSS/n) + k * ln(n)
    bic = n * np.log(rss / n) + k * np.log(n) if rss > 0 else -np.inf
    mae = mean_absolute_error(y_true, y_pred)
    return chi2_dof, bic, mae

def polynomial_workflow(train, test):
    X_train = np.arange(len(train))
    y_train = train['inflation_rate'].values
    X_test = np.arange(len(train), len(train) + len(test))
    y_test = test['inflation_rate'].values
   
    results = []
    for order in range(1, 10):
        # Fit polynomial and get covariance matrix
        coeffs, cov = np.polyfit(X_train, y_train, deg=order, cov=True)
       
        # Predictions on test set
        p = np.poly1d(coeffs)
        y_pred = p(X_test)
       
        # Metrics
        num_params = order + 1
        chi2_dof, bic, mae = compute_metrics(y_test, y_pred, num_params)
       
        results.append({
            'order': order,
            'coeffs': coeffs,
            'cov': cov,
            'chi2_dof': chi2_dof,
            'bic': bic,
            'mae': mae,
            'preds': y_pred
        })
   
    #identify the best polynomial model based on lowest mean absolute error
    best_poly = min(results, key=lambda x: x['mae'])
    return results, best_poly

#define function for ARIMA model
def arima_workflow(train, test, order=(1,1,1)):
    # Fit ARIMA (1,1,1)
    model = ARIMA(train['inflation_rate'], order=order).fit()
    preds = model.forecast(steps=len(test))
    mae = mean_absolute_error(test['inflation_rate'], preds)
    return preds, mae

# Execute
train, test = load_and_split_data("data\processed\inflation.csv")

poly_all_results, best_poly = polynomial_workflow(train, test)

arima_preds, arima_mae = arima_workflow(train, test)

# Identify best overall model
if best_poly['mae'] < arima_mae:
    best_overall_name = f"Polynomial (Order {best_poly['order']})"
    best_overall_mae = best_poly['mae']
else:
    best_overall_name = "ARIMA"
    best_overall_mae = arima_mae

# Plotting the polynomials against actual data
plt.plot(train['date'], train['inflation_rate'], label='Train Data', color='gray', alpha=0.4)
plt.plot(test['date'], test['inflation_rate'], label='Actual Test Data', color='black', linewidth=2)

for res in poly_all_results:
    # Highlighting the linear model (Order 1) with a thicker line
    lw = 2.5 if res['order'] == 1 else 1.0
    plt.plot(test['date'], res['preds'], label=f"Poly Order {res['order']}", linewidth=lw)

plt.xlabel('Date')
plt.ylabel('Inflation Rate (%)')
plt.title('Polynomial Model Performance Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('polynomial_comparison.png')

# Print outputs
print(f"Best Polynomial Model Details")
print(f"Order: {best_poly['order']}")
print(f"Coefficients: {best_poly['coeffs']}")
print("Covariance Matrix of Coefficients:")
print(best_poly['cov'])
print(f"Chi-squared per degree of freedom (test): {best_poly['chi2_dof']:.4f}")
print(f"BIC (test): {best_poly['bic']:.4f}")
print(f"MAE (test): {best_poly['mae']:.4f}")

print(f"\nARIMA Model")
print(f"Forecast MAE (test): {arima_mae:.4f}")

print("")
print(f"The best overall model is {best_overall_name} with a MAE of {best_overall_mae:.4f}")
