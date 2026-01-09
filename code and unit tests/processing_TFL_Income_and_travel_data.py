import re
from pathlib import Path
import pandas as pd

#define filepaths
TRENDS_RAW_PATH = Path("data/raw/tfl-transport-trends-data.xls")
INCOME_RAW_PATH = Path("data/raw/tfl income statement.xlsx")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRENDS_OUT_PATH = PROCESSED_DIR / "tfl_public_transport_journey_stages_annual.csv"
INCOME_OUT_PATH = PROCESSED_DIR / "tfl_income_annual.csv"

#Headers start on Excel row 5 so header=4
# Year + total (public transport journey stages, millions)

tfl_trends_sheet4_df = pd.read_excel(
    TRENDS_RAW_PATH,
    sheet_name="4",
    header=4,
    usecols="A:G")

# Clean trailing spaces in headers
tfl_trends_sheet4_df.columns = tfl_trends_sheet4_df.columns.str.strip()

tfl_public_transport_journeys_annual_df = (
    tfl_trends_sheet4_df[["Year", "Total"]]
    .rename(columns={
        "Year": "financial_year",
        "Total": "public_transport_journey_stages_m"
    })
    .dropna(subset=["financial_year"])
)

# Save to file (clear where it is stored)
tfl_public_transport_journeys_annual_df.to_csv(TRENDS_OUT_PATH, index=False)

print(f"DataFrame name: tfl_public_transport_journeys_annual_df")
print(f"Saved file: {TRENDS_OUT_PATH.resolve()}")
print(tfl_public_transport_journeys_annual_df.head(), "\n")


#TfL Income Statement

INCOME_SHEET = "0028 Income Statement with foot"

# Read with no header so we can detect where headers actually are
income_raw_noheader = pd.read_excel(
    INCOME_RAW_PATH,
    sheet_name=INCOME_SHEET,
    header=None
)

#detect header row by looking for multiple "YYYY/YY" patterns in a row to increase robustness
fy_pattern = re.compile(r"^\d{4}/\d{2}$")

header_row_idx = None
for i in range(0, min(25, len(income_raw_noheader))):
    row_vals = income_raw_noheader.iloc[i].astype(str).str.strip().tolist()
    fy_hits = sum(1 for v in row_vals if fy_pattern.match(v))
    if fy_hits >= 3: # threshold: row looks like a financial year header row
        header_row_idx = i
        break

if header_row_idx is None:
    raise ValueError(
        "Could not detect the financial year header row in the income statement. "
        "Open the sheet and check where the '2015/16'-style headers start."
    )

# Re-read using detected header row
income_df = pd.read_excel(
    INCOME_RAW_PATH,
    sheet_name=INCOME_SHEET,
    header=header_row_idx
)

#first column is the metric labels
metric_col = income_df.columns[0]
income_df = income_df.rename(columns={metric_col: "metric"})
income_df["metric"] = income_df["metric"].astype(str).str.strip()

#identify which columns are financial years
fy_cols = [c for c in income_df.columns if isinstance(c, str) and fy_pattern.match(c.strip())]

if not fy_cols:
    raise ValueError(
        "Detected income statement header row, but couldn't find any financial-year columns like '2015/16'."
    )

#extract key metrics
#atleast need Passenger income
passenger_row = income_df[income_df["metric"].str.contains("Passenger income", case=False, na=False)]
if passenger_row.empty:
    raise ValueError("Could not find a 'Passenger income' row in the income statement sheet.")

#Total income (attempt)
total_row = income_df[income_df["metric"].str.contains("Total", case=False, na=False)]

#tidy up output
tfl_income_annual_df = pd.DataFrame({
    "financial_year": fy_cols,
    "passenger_income_m": passenger_row.iloc[0][fy_cols].values,
})


if len(total_row) == 1:
    tfl_income_annual_df["total_income_m"] = total_row.iloc[0][fy_cols].values

# Save to file
tfl_income_annual_df.to_csv(INCOME_OUT_PATH, index=False)

print("Income Statement extracted")
print(f"DataFrame name: tfl_income_annual_df")
print(f"Saved file: {INCOME_OUT_PATH.resolve()}")
#print to check the dataframe  lloks as expected
print(tfl_income_annual_df.head(), "\n")


# Summary
print("STORED OUTPUTS")
print(f"1) {TRENDS_OUT_PATH} (from {TRENDS_RAW_PATH.name}, sheet '4')")
print(f"2) {INCOME_OUT_PATH} (from {INCOME_RAW_PATH.name}, sheet '{INCOME_SHEET}')")