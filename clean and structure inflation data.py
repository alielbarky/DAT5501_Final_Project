import pandas as pd
#load data and drop metadata rows
raw = pd.read_excel("data\raw\consumerpriceinflationdetailedreferencetables.xlsx", sheet_name="Table 37", header=None)
HEADER_ROW = 6
raw = raw.iloc[HEADER_ROW:].reset_index(drop=True)

# --- Step 3: Promote first row to header ---
raw.columns = raw.iloc[0]
df = raw.iloc[1:].reset_index(drop=True)

print(df)

# --- Step 4: Keep only required columns ---
df = df[["name", "CPIH ALL ITEMS"]].copy()
df.rename(
    columns={
        "name": "date",
        "CPIH ALL ITEMS": "cpih_index"
    },
    inplace=True
)

# --- Step 5: Convert types ---
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["cpih_index"] = pd.to_numeric(df["cpih_index"], errors="coerce")

# --- Step 6: Sort chronologically ---
df = df.sort_values("date")

# --- Step 7: Calculate year-on-year inflation (%) ---
df["inflation_rate"] = (
    df["cpih_index"] / df["cpih_index"].shift(12) - 1
) * 100

# --- Step 8: Final output ---
output = df[["date", "inflation_rate"]].dropna()

# --- Step 9: Save ---
output.to_csv("inflation.csv", index=False)

print("inflation.csv created successfully")