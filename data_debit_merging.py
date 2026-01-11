import pandas as pd
from functools import reduce
from pathlib import Path
from collections import defaultdict



folder = Path("Data_debit/raw_data_individual")
files_raw_debit = list(folder.glob("*.csv"))



def load_one_csv_data_debit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1, sep=";")
    dataset_name = path.stem[:5]
    print(f"working on {dataset_name}")
    df = df.drop(columns=["Excel Date"])
    df = df.rename(columns={"Debit []": dataset_name})
    df = df.rename(columns={"Debit [m3/s]": dataset_name})
    print("Columns in", path, ":", df.columns.tolist())
    return df


dfs = [load_one_csv_data_debit(f) for f in files_raw_debit]

Data_debit_raw_merge = reduce(lambda left, right: pd.merge(left, right, on=["ISO Date", "Unix Date"], how="outer"),dfs)


station_cols = [c for c in Data_debit_raw_merge.columns if c not in ["ISO Date", "Unix Date"]]
groups = defaultdict(list)
for col in station_cols:
    base = col[:3]
    groups[base].append(col)

for base, cols in groups.items():
    if len(cols) == 1:
        if cols[0] != base:
            Data_debit_raw_merge.rename(columns={cols[0]: base}, inplace=True)
    else:
        Data_debit_raw_merge[base] = Data_debit_raw_merge[cols].bfill(axis=1).iloc[:, 0]
        for c in cols:
            if c != base:
                Data_debit_raw_merge.drop(columns=c, inplace=True)


Data_debit_raw_merge = Data_debit_raw_merge.reindex(columns=Data_debit_raw_merge.columns)

Data_debit_raw_merge = Data_debit_raw_merge.replace(["", " ", "NA", "N/A", "NaN"], pd.NA)    

Data_debit_raw_merge.to_csv("C:/Unil/Master/Automne 2/Ada/Projet/Data_debit/Data_debit_raw_merge.csv",index=False)
print("Data_debit_raw_merge.csv is created")


