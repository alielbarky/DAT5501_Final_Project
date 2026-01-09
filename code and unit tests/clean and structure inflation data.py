import pandas as pd

#load data and drop useless rows
raw = pd.read_excel("data\raw\consumerpriceinflationdetailedreferencetables.xlsx", sheet_name="Table 37", header=None)
HEADER_ROW = 6
raw = raw.iloc[HEADER_ROW:].reset_index(drop=True)
raw.columns = raw.iloc[0]
df = raw.iloc[1:].reset_index(drop=True)

#check that the dataframe is as expected
print(df)

#filter for the columns we need
df = df[["name", "CPIH ALL ITEMS"]].copy()
df.rename(
    columns={
        "name": "date",
        "CPIH ALL ITEMS": "cpih_index"
    },
    inplace=True
)

#make sure data types are corrct
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["cpih_index"] = pd.to_numeric(df["cpih_index"], errors="coerce")

#sort by date
df = df.sort_values("date")

#year on year inflation
df["inflation_rate"] = (
    df["cpih_index"] / df["cpih_index"].shift(12) - 1) * 100

#final outputs
output = df[["date", "inflation_rate"]].dropna()

#save the file
output.to_csv("inflation.csv", index=False)

print("inflation.csv created successfully")