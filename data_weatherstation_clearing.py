import pandas as pd
from pathlib import Path
from functools import reduce

# Input folder
folder = r"C:\Unil\Master\Automne 2\Ada\Projet\Data_meteo\merged"
folder_path = Path(folder)

# Output file
output_file = r"C:\Unil\Master\Automne 2\Ada\Projet\Data_meteo\data_precipitation_merge_raw.csv"

# Columns to keep from each file
cols_to_keep = ["reference_timestamp", "rre150h0"]

dfs = []

for csv_file in sorted(folder_path.glob("*.csv")):
    print(f"Loading: {csv_file.name}")
    
    # Read CSV with ; as separator
    df = pd.read_csv(csv_file, sep=";")
    
    # Check required columns
    missing = [c for c in cols_to_keep if c not in df.columns]
    if missing:
        print(f"  ⚠ Skipping {csv_file.name}: missing columns {missing}")
        continue
    
    # Keep only the two columns
    df = df[cols_to_keep].copy()
    
    # ---- Convert reference_timestamp text to ISO format and rename ----
    # "2000-01-01 04:00:00" -> "2000-01-01T04:00:00+01:00"
    df["reference_timestamp"] = (
        df["reference_timestamp"]
        .astype(str)
        .str.replace(" ", "T", regex=False)  # add the 'T'
        + "+01:00"                            # add timezone offset
    )
    
    # Rename time column to match your other dataframe
    df = df.rename(columns={"reference_timestamp": "ISO Date"})
    # -------------------------------------------------------------------
    
    # Rename rre150h0 → file name (without .csv)
    new_col_name = csv_file.stem
    df = df.rename(columns={"rre150h0": new_col_name})
    
    dfs.append(df)

# Merge all DataFrames on ISO Date
if dfs:
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="ISO Date", how="outer"),
        dfs
    )
    
    # Sort chronologically
    merged_df = merged_df.sort_values("ISO Date").reset_index(drop=True)
    
    
    # ---------------------------------------------------------
    # 🔧 Add Unix timestamp column
    # ---------------------------------------------------------
    # Convert ISO string → datetime (timezone-aware)
    merged_df["Unix Date"] = pd.to_datetime(
        merged_df["ISO Date"],
        format="%Y-%m-%dT%H:%M:%S%z",
        errors="coerce"
    ).astype("int64") // 10**9
    # ---------------------------------------------------------
    
    
    # Save to file
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Merged file saved at:\n{output_file}")
else:
    print("❌ No valid CSV files to merge.")