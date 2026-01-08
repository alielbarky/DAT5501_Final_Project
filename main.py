import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# =============================
# LOAD DATA
# =============================
DATA = Path("data/processed")

income = pd.read_csv(DATA / "tfl_income_annual.csv")
inflation_hist = pd.read_csv(DATA / "inflation.csv", parse_dates=["date"])
inflation_proj = pd.read_csv(DATA / "projected_inflation.csv", parse_dates=["date"])

# =============================
# CLEAN & COMBINE MONTHLY INFLATION
# =============================

# Historical: keep up to Nov 2025
inflation_hist = inflation_hist[inflation_hist["date"] <= pd.Timestamp("2025-11-30")].copy()
inflation_hist["inflation"] = inflation_hist["inflation_rate"]

# Projected: keep from Apr 2026 onward (discard Dec 2025–Mar 2026)
inflation_proj = inflation_proj[inflation_proj["date"] >= pd.Timestamp("2026-04-01")].copy()
inflation_proj["inflation"] = inflation_proj["predicted_inflation"]

# Combine monthly inflation
inflation_all = pd.concat(
    [inflation_hist[["date", "inflation"]], inflation_proj[["date", "inflation"]]],
    ignore_index=True,
)

# =============================
# MONTHLY → FINANCIAL YEAR (ANNUAL AVG INFLATION)
# =============================
fy_start = inflation_all["date"].dt.year.where(
    inflation_all["date"].dt.month >= 4,
    inflation_all["date"].dt.year - 1,
)

inflation_all["financial_year"] = (
    fy_start.astype(str) + "/" + (fy_start + 1).astype(str).str[-2:]
)

annual_inflation = (
    inflation_all.groupby("financial_year")["inflation"]
    .mean()
    .reset_index()
    .rename(columns={"inflation": "avg_inflation"})
)

# =============================
# REAL INCOME (HISTORICAL INCOME YEARS)
# =============================
income["financial_year"] = income["financial_year"].astype(str).str.strip()

df = income.merge(annual_inflation, on="financial_year", how="left")

df["inflation_factor"] = 1 + df["avg_inflation"] / 100

df["price_index"] = df["inflation_factor"].cumprod()
# Rebase to first income year = 100
df["price_index"] = 100 * df["price_index"] / df["price_index"].iloc[0]

df["real_passenger_income_m"] = df["passenger_income_m"] * 100 / df["price_index"]

# =============================
# EXTEND TO 2030/31 (NOMINAL INCOME STAGNATES)
# =============================

def fy_start_year(fy: str) -> int:
    return int(str(fy).split("/")[0])

# Add sortable start-year key
annual_inflation["fy_start"] = annual_inflation["financial_year"].apply(fy_start_year)
df["fy_start"] = df["financial_year"].apply(fy_start_year)

last_income_fy_start = int(df["fy_start"].max())

# Only future FY inflation beyond last income year, and only up to 2030/31
future_fy_infl = annual_inflation[(annual_inflation["fy_start"] > last_income_fy_start) & (annual_inflation["fy_start"] <= 2030)].copy()
future_fy_infl = future_fy_infl.sort_values("fy_start")

last_nominal_income = float(df["passenger_income_m"].iloc[-1])
current_price_index = float(df["price_index"].iloc[-1])

rows = []
for _, row in future_fy_infl.iterrows():
    inflation_factor = 1 + float(row["avg_inflation"]) / 100
    current_price_index *= inflation_factor

    rows.append(
        {
            "financial_year": row["financial_year"],
            "passenger_income_m": last_nominal_income,
            "avg_inflation": float(row["avg_inflation"]),
            "inflation_factor": inflation_factor,
            "price_index": current_price_index,
            "real_passenger_income_m": last_nominal_income * 100 / current_price_index,
            "fy_start": int(row["fy_start"]),
        }
    )

df_future = pd.DataFrame(rows)

df_all = pd.concat([df, df_future], ignore_index=True)
df_all = df_all.sort_values("fy_start").reset_index(drop=True)

# =============================
# EXPORTS + HEADLINE METRICS (FOR REPORT)
# =============================
OUTPUT_DIR = DATA / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Save the modelling output table
OUT_TABLE = OUTPUT_DIR / "tfl_real_income_extended_2015_2031.csv"
df_all.to_csv(OUT_TABLE, index=False)

# 2) Headline metrics
first_year = df_all.iloc[0]
last_observed = df_all[df_all["fy_start"] == last_income_fy_start].iloc[-1]
last_projected = df_all.iloc[-1]


def pct_change(a, b):
    return (b / a - 1) * 100


headline = {
    "first_financial_year": first_year["financial_year"],
    "last_observed_financial_year": last_observed["financial_year"],
    "last_projected_financial_year": last_projected["financial_year"],
    "real_income_first_m": float(first_year["real_passenger_income_m"]),
    "real_income_last_observed_m": float(last_observed["real_passenger_income_m"]),
    "real_income_last_projected_m": float(last_projected["real_passenger_income_m"]),
    "pct_change_first_to_last_observed": float(
        pct_change(first_year["real_passenger_income_m"], last_observed["real_passenger_income_m"])
    ),
    "pct_change_last_observed_to_last_projected": float(
        pct_change(last_observed["real_passenger_income_m"], last_projected["real_passenger_income_m"])
    ),
    "pct_change_first_to_last_projected": float(
        pct_change(first_year["real_passenger_income_m"], last_projected["real_passenger_income_m"])
    ),
}

headline_df = pd.DataFrame([headline])
OUT_HEADLINE = OUTPUT_DIR / "headline_metrics.csv"
headline_df.to_csv(OUT_HEADLINE, index=False)

print("Saved outputs:")
print("-", OUT_TABLE)
print("-", OUT_HEADLINE)
print("Headline metrics:")
print(headline_df.T)

# =============================
# KEY FIGURE (SAVED ONCE, NO DUPLICATION)
# =============================
FIG_PATH = OUTPUT_DIR / "figure_1_real_passenger_income.png"

plt.figure(figsize=(10, 6))
plt.plot(
    df_all["financial_year"],
    df_all["real_passenger_income_m"],
    linewidth=2,
    label="Real passenger income (£m)",
)

# Projection boundary line (at last observed income FY)
boundary_idx = df_all.index[df_all["fy_start"] == last_income_fy_start][0]
plt.axvline(x=boundary_idx, linestyle="--", linewidth=1, label="Start of projection")

plt.xticks(rotation=45)
plt.xlabel("Financial Year")
plt.ylabel("Real Passenger Income (£m)")
plt.title("TfL Real Passenger Income (Extended with Projected Inflation; Nominal Income Stagnates)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)

print("Saved figure:")
print("-", FIG_PATH)

# =============================
# NEXT STEP: SENSITIVITY ANALYSIS (INFLATION SHOCK) + SUPPLEMENTAL FIGURE
# =============================
# Purpose: demonstrate result stability (S53) by testing how real income changes
# under +/- 1 percentage point inflation in future years.

# Build scenario paths only for projected years (beyond last observed income FY)
base_future = future_fy_infl.copy()

# Scenario definitions (annual inflation shock applied to the projected FY averages)
scenarios = {
    "baseline": 0.0,
    "low_inflation_-1pp": -1.0,
    "high_inflation_+1pp": 1.0,
}

scenario_rows = []

# Seed from last observed year
seed_price_index = float(df_all[df_all["fy_start"] == last_income_fy_start]["price_index"].iloc[-1])
seed_nominal_income = float(df_all[df_all["fy_start"] == last_income_fy_start]["passenger_income_m"].iloc[-1])

for scen_name, shock in scenarios.items():
    current_pi = seed_price_index

    for _, r in base_future.iterrows():
        adj_infl = float(r["avg_inflation"]) + shock
        infl_factor = 1 + adj_infl / 100.0
        current_pi *= infl_factor

        scenario_rows.append(
            {
                "scenario": scen_name,
                "financial_year": r["financial_year"],
                "fy_start": int(r["fy_start"]),
                "avg_inflation": float(r["avg_inflation"]),
                "inflation_shock_pp": float(shock),
                "adj_avg_inflation": float(adj_infl),
                "price_index": float(current_pi),
                "passenger_income_m": float(seed_nominal_income),
                "real_passenger_income_m": float(seed_nominal_income) * 100.0 / float(current_pi),
            }
        )

scenario_df = pd.DataFrame(scenario_rows)

# Export sensitivity table
OUT_SENS = OUTPUT_DIR / "sensitivity_inflation_shock.csv"
scenario_df.to_csv(OUT_SENS, index=False)
print("Saved sensitivity table:")
print("-", OUT_SENS)

# Plot supplemental figure: baseline vs +/- 1pp inflation (projected years only)
FIG2_PATH = OUTPUT_DIR / "figure_2_sensitivity_inflation_shock.png"

plt.figure(figsize=(10, 6))

# Baseline full series from df_all
plt.plot(
    df_all["financial_year"],
    df_all["real_passenger_income_m"],
    linewidth=2,
    label="Baseline (real income)",
)

# Overlay scenario lines for projected years
for scen_name in ["low_inflation_-1pp", "high_inflation_+1pp"]:
    sub = scenario_df[scenario_df["scenario"] == scen_name].sort_values("fy_start")
    plt.plot(
        sub["financial_year"],
        sub["real_passenger_income_m"],
        linestyle="--",
        linewidth=2,
        label=scen_name.replace("_", " "),
    )

# Projection boundary
boundary_idx = df_all.index[df_all["fy_start"] == last_income_fy_start][0]
plt.axvline(x=boundary_idx, linestyle="--", linewidth=1, label="Start of projection")

plt.xticks(rotation=45)
plt.xlabel("Financial Year")
plt.ylabel("Real Passenger Income (£m)")
plt.title("Sensitivity: Real Income Under +/-1pp Inflation Shock (Nominal Income Stagnates)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG2_PATH, dpi=300)

print("Saved supplemental figure:")
print("-", FIG2_PATH)

# =============================
# NEXT STEP: MODEL VALIDATION & DIAGNOSTICS (ROBUSTNESS CHECKS)
# =============================
# Purpose: demonstrate stability and transparency of results for assessment criteria.

# 1) Check for missing values and basic integrity
integrity_checks = {
    "rows_total": int(len(df_all)),
    "missing_avg_inflation": int(df_all["avg_inflation"].isna().sum()) if "avg_inflation" in df_all.columns else 0,
    "missing_price_index": int(df_all["price_index"].isna().sum()),
    "missing_real_income": int(df_all["real_passenger_income_m"].isna().sum()),
}

integrity_df = pd.DataFrame([integrity_checks])
OUT_INTEGRITY = OUTPUT_DIR / "integrity_checks.csv"
integrity_df.to_csv(OUT_INTEGRITY, index=False)

print("Integrity checks:")
print(integrity_df.T)
print("Saved integrity checks:")
print("-", OUT_INTEGRITY)

# 2) Alternative rebasing year (robustness test)
# Rebase price index to a mid-series year (e.g. 2019/20) and recompute real income
rebase_year = "2019/20"

if rebase_year in set(df_all["financial_year"]):
    base_pi = float(df_all.loc[df_all["financial_year"] == rebase_year, "price_index"].iloc[0])

    df_rebased = df_all.copy()
    df_rebased["price_index_rebased"] = 100.0 * df_rebased["price_index"] / base_pi
    df_rebased["real_income_rebased_m"] = (
        df_rebased["passenger_income_m"] * 100.0 / df_rebased["price_index_rebased"]
    )

    OUT_REBASE = OUTPUT_DIR / "rebased_real_income_2019_20.csv"
    df_rebased[[
        "financial_year",
        "price_index_rebased",
        "real_income_rebased_m",
    ]].to_csv(OUT_REBASE, index=False)

    print(f"Rebased outputs saved using base year {rebase_year}:")
    print("-", OUT_REBASE)

# 3) Simple trend diagnostics (rolling mean)
# Helps illustrate structural change without adding complexity

df_all_sorted = df_all.sort_values("fy_start").reset_index(drop=True)
df_all_sorted["real_income_3yr_ma"] = df_all_sorted["real_passenger_income_m"].rolling(window=3, min_periods=1).mean()

OUT_TREND = OUTPUT_DIR / "real_income_rolling_mean.csv"
df_all_sorted[[
    "financial_year",
    "real_passenger_income_m",
    "real_income_3yr_ma",
]].to_csv(OUT_TREND, index=False)

print("Saved trend diagnostics:")
print("-", OUT_TREND)