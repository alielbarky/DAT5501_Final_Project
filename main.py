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
inflation_hist = inflation_hist[
    inflation_hist["date"] <= pd.Timestamp("2025-11-30")
].copy()

inflation_hist["inflation"] = inflation_hist["inflation_rate"]

# Projected: keep from Apr 2026 onward
inflation_proj = inflation_proj[
    inflation_proj["date"] >= pd.Timestamp("2026-04-01")
].copy()

inflation_proj["inflation"] = inflation_proj["predicted_inflation"]

# Combine monthly inflation
inflation_all = pd.concat(
    [inflation_hist[["date", "inflation"]],
     inflation_proj[["date", "inflation"]]],
    ignore_index=True
)

# =============================
# MONTHLY → FINANCIAL YEAR
# =============================
fy_start = inflation_all["date"].dt.year.where(
    inflation_all["date"].dt.month >= 4,
    inflation_all["date"].dt.year - 1
)

inflation_all["financial_year"] = (
    fy_start.astype(str)
    + "/"
    + (fy_start + 1).astype(str).str[-2:]
)

annual_inflation = (
    inflation_all
    .groupby("financial_year")["inflation"]
    .mean()
    .reset_index()
    .rename(columns={"inflation": "avg_inflation"})
)

# =============================
# REAL INCOME (HISTORICAL)
# =============================
income["financial_year"] = income["financial_year"].str.strip()

df = income.merge(
    annual_inflation,
    on="financial_year",
    how="left"
)

df["inflation_factor"] = 1 + df["avg_inflation"] / 100
df["price_index"] = df["inflation_factor"].cumprod()
df["price_index"] = 100 * df["price_index"] / df["price_index"].iloc[0]

df["real_passenger_income_m"] = (
    df["passenger_income_m"] * 100 / df["price_index"]
)

# =============================
# EXTEND TO 2030/31 (STAGNATION)
# =============================
last_nominal_income = df["passenger_income_m"].iloc[-1]
last_price_index = df["price_index"].iloc[-1]

future_years = annual_inflation[
    ~annual_inflation["financial_year"].isin(df["financial_year"])
].sort_values("financial_year")

rows = []
current_price_index = last_price_index

for _, row in future_years.iterrows():
    inflation_factor = 1 + row["avg_inflation"] / 100
    current_price_index *= inflation_factor

    rows.append({
        "financial_year": row["financial_year"],
        "passenger_income_m": last_nominal_income,
        "price_index": current_price_index,
        "real_passenger_income_m": last_nominal_income * 100 / current_price_index
    })

df_future = pd.DataFrame(rows)

df_all = pd.concat([df, df_future], ignore_index=True)

# =============================
# KEY FIGURE: REAL PASSENGER INCOME (FIXED ORDER)
# =============================

# Build projected inflation -> financial year -> annual avg inflation
fy_start_proj = inflation_proj["date"].dt.year.where(
    inflation_proj["date"].dt.month >= 4,
    inflation_proj["date"].dt.year - 1
)

inflation_proj["financial_year"] = (
    fy_start_proj.astype(str)
    + "/"
    + (fy_start_proj + 1).astype(str).str[-2:]
)

proj_annual = (
    inflation_proj
    .groupby("financial_year")["predicted_inflation"]
    .mean()
    .reset_index()
    .rename(columns={"predicted_inflation": "avg_inflation"})
)

# Helper for chronological sorting/filtering
def fy_start_year(fy: str) -> int:
    return int(str(fy).split("/")[0])

df["fy_start"] = df["financial_year"].apply(fy_start_year)
proj_annual["fy_start"] = proj_annual["financial_year"].apply(fy_start_year)

last_income_fy_start = df["fy_start"].max()

# Keep only FYs after the last income year
future_years = proj_annual[proj_annual["fy_start"] > last_income_fy_start].copy()
future_years = future_years.sort_values("fy_start")

# Optional: drop FY 2031/32 if it appears (April 2031 creates it)
future_years = future_years[future_years["fy_start"] <= 2030]

# Extend price index and real income (nominal stagnates)
last_nominal_income = df["passenger_income_m"].iloc[-1]
current_price_index = df["price_index"].iloc[-1]

rows = []
for _, row in future_years.iterrows():
    inflation_factor = 1 + row["avg_inflation"] / 100
    current_price_index *= inflation_factor

    rows.append({
        "financial_year": row["financial_year"],
        "passenger_income_m": last_nominal_income,
        "price_index": current_price_index,
        "real_passenger_income_m": last_nominal_income * 100 / current_price_index,
        "fy_start": row["fy_start"],
    })

df_future = pd.DataFrame(rows)

df_all = pd.concat([df, df_future], ignore_index=True)
df_all = df_all.sort_values("fy_start").reset_index(drop=True)

# Plot key figure

plt.figure(figsize=(10, 6))

plt.plot(
    df_all["financial_year"],
    df_all["real_passenger_income_m"],
    linewidth=2,
    label="Real passenger income (£m)"
)

# Mark start of projection (first year after last income year)
proj_start_idx = df_all.index[df_all["fy_start"] == last_income_fy_start][0]
plt.axvline(x=proj_start_idx, linestyle="--", linewidth=1, label="Start of projection")

plt.xticks(rotation=45)
plt.xlabel("Financial Year")
plt.ylabel("Real Passenger Income (£m)")
plt.title("TfL Real Passenger Income (Extended with Projected Inflation; Nominal Income Stagnates)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('TFL real income projection.png')

#EXPORTS + HEADLINE METRICS (FOR REPORT)

# 1) Save the modelling output table for use in report/appendix
OUTPUT_DIR = DATA / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_all.to_csv(OUTPUT_DIR / "tfl_real_income_extended_2015_2031.csv", index=False)

# 2) Headline metrics (simple, report-ready)
# - Change from first observed year to last observed year
# - Change from last observed year to last projected year
# - Total change across full horizon

# Ensure chronological order for metrics
if "fy_start" not in df_all.columns:
    df_all["fy_start"] = df_all["financial_year"].apply(fy_start_year)

df_all_sorted = df_all.sort_values("fy_start").reset_index(drop=True)

first_year = df_all_sorted.iloc[0]
last_observed = df_all_sorted[df_all_sorted["fy_start"] == last_income_fy_start].iloc[-1]
last_projected = df_all_sorted.iloc[-1]

def pct_change(a, b):
    return (b / a - 1) * 100

headline = {
    "first_financial_year": first_year["financial_year"],
    "last_observed_financial_year": last_observed["financial_year"],
    "last_projected_financial_year": last_projected["financial_year"],
    "real_income_first_m": float(first_year["real_passenger_income_m"]),
    "real_income_last_observed_m": float(last_observed["real_passenger_income_m"]),
    "real_income_last_projected_m": float(last_projected["real_passenger_income_m"]),
    "pct_change_first_to_last_observed": float(pct_change(first_year["real_passenger_income_m"], last_observed["real_passenger_income_m"])),
    "pct_change_last_observed_to_last_projected": float(pct_change(last_observed["real_passenger_income_m"], last_projected["real_passenger_income_m"])),
    "pct_change_first_to_last_projected": float(pct_change(first_year["real_passenger_income_m"], last_projected["real_passenger_income_m"])),
}

headline_df = pd.DataFrame([headline])
headline_df.to_csv(OUTPUT_DIR / "headline_metrics.csv", index=False)

print("Saved outputs:")
print("-", OUTPUT_DIR / "tfl_real_income_extended_2015_2031.csv")
print("-", OUTPUT_DIR / "headline_metrics.csv")
print("Headline metrics:")
print(headline_df.T)

# 3) Save the key figure to file (for report)
FIG_PATH = OUTPUT_DIR / "figure_1_real_passenger_income.png"
plt.figure(figsize=(10, 6))
plt.plot(
    df_all_sorted["financial_year"],
    df_all_sorted["real_passenger_income_m"],
    linewidth=2,
    label="Real passenger income (£m)"
)

# Start of projection boundary
boundary_idx = df_all_sorted.index[df_all_sorted["fy_start"] == last_income_fy_start][0]
plt.axvline(x=boundary_idx, linestyle="--", linewidth=1, label="Start of projection")

plt.xticks(rotation=45)
plt.xlabel("Financial Year")
plt.ylabel("Real Passenger Income (£m)")
plt.title("TfL Real Passenger Income (Extended with Projected Inflation; Nominal Income Stagnates)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.show()
