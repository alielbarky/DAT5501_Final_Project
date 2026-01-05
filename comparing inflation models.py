import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    train = df[(df['date'] <= '2013-01-01')]
    test = df[(df['date'] > '2013-01-01') & (df['date'] <= '2019-01-01')]

    return train, test

def regression_model(train, test):
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['inflation_rate'].values

    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    y_test = test['inflation_rate'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return mae

def arima_model(train, test, order=(1,1,1)):
    model = ARIMA(train['inflation_rate'], order=order)
    fitted = model.fit()

    preds = fitted.forecast(steps=len(test))
    mae = mean_absolute_error(test['inflation_rate'], preds)

    return mae


def decision_tree_model(train, test, max_depth=5):
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['inflation_rate'].values

    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    y_test = test['inflation_rate'].values

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return mae

def evaluate_models(csv_path):
    train, test = load_and_split_data(csv_path)

    results = {
        "Regression": regression_model(train, test),
        "ARIMA": arima_model(train, test),
        "Decision Tree": decision_tree_model(train, test)
    }

    best_model = min(results, key=results.get)

    return results, best_model

results, best_model = evaluate_models("inflation.csv")

print("Model MAE scores:")
for model, score in results.items():
    print(f"{model}: {score:.3f}")

print(f"\n the best model is {best_model}")