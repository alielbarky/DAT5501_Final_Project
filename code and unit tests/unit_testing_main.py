import pandas as pd
import numpy as np

#functions replicated from main.py for testability

def fy_start_year(fy: str) -> int:
    #Extract start year from financial year string
    return int(str(fy).split("/")[0])


def pct_change(a, b):
    #Percentage change from a to b
    return (b / a - 1) * 100

#the actual tests:

def test_fy_start_year_basic():
    assert fy_start_year("2019/20") == 2019
    assert fy_start_year("2015/16") == 2015


def test_fy_start_year_string_coercion():
    assert fy_start_year(2020) == 2020


def test_pct_change_positive_growth():
    assert np.isclose(pct_change(100, 110), 10.0)


def test_pct_change_negative_growth():
    assert np.isclose(pct_change(200, 150), -25.0)


def test_pct_change_zero_baseline():
    #division by zero should raise an error
    try:
        pct_change(0, 100)
        assert False, "Expected ZeroDivisionError"
    except ZeroDivisionError:
        assert True


#inflation aggregation logic

def test_financial_year_assignment():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2020-03-31", # FY 2019/20
            "2020-04-01", # FY 2020/21
        ]),
        "inflation": [2.0, 3.0],
    })

    fy_start = df["date"].dt.year.where(
        df["date"].dt.month >= 4,
        df["date"].dt.year - 1,
    )

    df["financial_year"] = (
        fy_start.astype(str) + "/" + (fy_start + 1).astype(str).str[-2:]
    )

    assert df.loc[0, "financial_year"] == "2019/20"
    assert df.loc[1, "financial_year"] == "2020/21"


def test_annual_inflation_mean():
    df = pd.DataFrame({
        "financial_year": ["2020/21", "2020/21", "2021/22"],
        "inflation": [2.0, 4.0, 3.0],
    })

    annual = (
        df.groupby("financial_year")["inflation"]
        .mean()
        .reset_index()
    )

    assert np.isclose(
        annual.loc[annual["financial_year"] == "2020/21", "inflation"].iloc[0],
        3.0,
    )

# Real income calculations

def test_price_index_cumprod_and_rebasing():
    df = pd.DataFrame({
        "avg_inflation": [2.0, 3.0, 5.0]
    })

    df["inflation_factor"] = 1 + df["avg_inflation"] / 100
    df["price_index"] = df["inflation_factor"].cumprod()
    df["price_index"] = 100 * df["price_index"] / df["price_index"].iloc[0]

    assert np.isclose(df.loc[0, "price_index"], 100.0)
    assert df.loc[1, "price_index"] > df.loc[0, "price_index"]
    assert df.loc[2, "price_index"] > df.loc[1, "price_index"]


def test_real_income_deflation():
    df = pd.DataFrame({
        "passenger_income_m": [1000, 1000],
        "price_index": [100, 110],
    })

    df["real_passenger_income_m"] = df["passenger_income_m"] * 100 / df["price_index"]

    assert np.isclose(df.loc[0, "real_passenger_income_m"], 1000.0)
    assert np.isclose(df.loc[1, "real_passenger_income_m"], 909.09, atol=0.1)


#future extension logic

def test_future_real_income_declines_with_positive_inflation():
    seed_price_index = 100.0
    seed_income = 1000.0

    future_inflation = [2.0, 3.0]
    real_values = []

    current_pi = seed_price_index
    for infl in future_inflation:
        current_pi *= (1 + infl / 100)
        real_values.append(seed_income * 100 / current_pi)

    assert real_values[1] < real_values[0]


#sensitivity analysis logic testing

def test_inflation_shock_effect_direction():
    seed_price_index = 100.0
    income = 1000.0
    base_infl = 3.0

    pi_low = seed_price_index * (1 + (base_infl - 1) / 100)
    pi_high = seed_price_index * (1 + (base_infl + 1) / 100)

    real_low = income * 100 / pi_low
    real_high = income * 100 / pi_high

    assert real_low > real_high


#data integrity checks

def test_integrity_check_no_missing():
    df = pd.DataFrame({
        "avg_inflation": [2.0, 3.0],
        "price_index": [100.0, 103.0],
        "real_passenger_income_m": [1000.0, 970.0],
    })

    assert df.isna().sum().sum() == 0