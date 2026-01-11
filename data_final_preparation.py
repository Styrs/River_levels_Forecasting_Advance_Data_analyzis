import pandas as pd
import os
import re
import data_clearing_function
import numpy as np


##############################################
# Import cleaned data
##############################################

data_precipitation_clean_merge = pd.read_csv(
    "Data_meteo/data_precipitation_clean_merge.csv", encoding="utf-8"
)
data_debit_clean_merge = pd.read_csv(
    "Data_debit/data_debit_clean_merge.csv", encoding="utf-8"
)

data_water_station_period_summary = pd.read_csv(
    "Data_debit/water_station_period_summary.csv", encoding="utf-8"
)

###############################################
# Helper to extract linked station list; make the list of stations in the right format
###############################################


def convert_cell_to_station_list(value):
    """
    Turn a cell from matching_* columns into a Python list of station names.

    Handles:
    - NaN → []
    - comma-separated strings: "S1,S2, S3"
    - already-a-list → returned as is
    """
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # split on comma, strip spaces, drop empty pieces
        return [s.strip() for s in value.split(",") if s.strip()]
    # Fallback: unknown type → empty
    return []


###############################################
# Feature-engineering functions; adding sum for precipitation, mean for debit and time features
###############################################

def add_weather_rolling_precip(df: pd.DataFrame,
                               windows_days=(14, 30)) -> pd.DataFrame:
    """
    For each weather station column (prefixed with 'W_'), add rolling
    precipitation sums over windows of 'windows_days' days.

    Uses time-based rolling with Unix Date → datetime index.
    """
    if "Unix Date" not in df.columns:
        raise KeyError("Column 'Unix Date' is required in df.")

    # Ensure time is sorted
    df = df.sort_values("Unix Date").copy()

    # Create a datetime index from Unix timestamp
    dt_index = pd.to_datetime(df["Unix Date"], unit="s")

    # We'll apply rolling on a temporary copy with datetime index
    temp = df.copy()
    temp.index = dt_index

    weather_cols = [c for c in df.columns if c.startswith("W_")]

    for col in weather_cols:
        for d in windows_days:
            new_col = f"{col}_sum_{d}d"
            # Time-based rolling window: last d days
            df[new_col] = temp[col].rolling(f"{d}D").sum().values

    return df


def add_water_rolling_debit(df: pd.DataFrame,
                            water_station_columns,
                            windows_days=(14, 30)) -> pd.DataFrame:
    """
    For each water station column in 'water_station_columns',
    add rolling mean of debit over windows of 'windows_days' days.
    """
    if "Unix Date" not in df.columns:
        raise KeyError("Column 'Unix Date' is required in df.")

    df = df.sort_values("Unix Date").copy()
    dt_index = pd.to_datetime(df["Unix Date"], unit="s")

    temp = df.copy()
    temp.index = dt_index

    # Keep only those water station columns that actually exist in df
    water_cols = [c for c in water_station_columns if c in df.columns]

    for col in water_cols:
        for d in windows_days:
            new_col = f"{col}_mean_{d}d"
            df[new_col] = temp[col].rolling(f"{d}D").mean().values

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features:
    - 'week_of_year': ISO week number derived from Unix Date
    - 'sin_week', 'cos_week': cyclical encoding for seasonality
    """
    if "Unix Date" not in df.columns:
        raise KeyError("Column 'Unix Date' is required in df.")

    df = df.copy()

    # Convert Unix timestamp → datetime
    dt = pd.to_datetime(df["Unix Date"], unit="s")

    # ISO week number (Python-level, compatible with all pandas versions)
    df["week_of_year"] = dt.apply(lambda x: x.isocalendar().week)

    # Cyclical encoding
    # Using 52 weeks as the cycle (standard convention)
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df

###############################################
# Build base per-station dataframe 
###############################################

def prepare_single_station(row,
                           data_precipitation_clean_merge,
                           data_debit_clean_merge):
    """
    Build the *base* dataset for a single water station, based on one row
    of data_water_station_period_summary.

    This function:
    - selects the correct time window
    - adds the main water station
    - adds matching weather stations (prefixed with 'W_')
    - adds matching other water stations (no renaming)

    It does NOT add rolling features or time indicators.
    """

    main_station = row["Water station"]
    start_unix = row["start_time"]
    end_unix = row["end_time"]

    # 1. Base: time window + main station
    mask_debit = (
        (data_debit_clean_merge["Unix Date"] >= start_unix) &
        (data_debit_clean_merge["Unix Date"] <= end_unix)
    )
    debit_window = data_debit_clean_merge.loc[mask_debit]

    if main_station not in debit_window.columns:
        raise KeyError(f"Main water station '{main_station}' not found!")

    df_station = debit_window[["ISO Date", "Unix Date", main_station]].copy()
    df_station = df_station.rename(columns={main_station: "self_data"})

    # 2. Add matching WEATHER stations with W_ prefix
    weather_stations = convert_cell_to_station_list(row.get("matching_weather_stations"))
    if weather_stations:
        mask_precip = (
            (data_precipitation_clean_merge["Unix Date"] >= start_unix) &
            (data_precipitation_clean_merge["Unix Date"] <= end_unix)
        )
        precip_window = data_precipitation_clean_merge.loc[mask_precip]

        for ws in weather_stations:
            if ws not in precip_window.columns:
                continue

            renamed = f"W_{ws}"

            to_merge = precip_window[["Unix Date", ws]].copy()
            to_merge = to_merge.rename(columns={ws: renamed})

            df_station = df_station.merge(to_merge, on="Unix Date", how="left")

    # 3. Add matching WATER stations (no renaming)
    other_water_stations = convert_cell_to_station_list(row.get("matching_water_stations"))
    if other_water_stations:
        for ws in other_water_stations:
            if ws not in debit_window.columns:
                continue

            to_merge = debit_window[["Unix Date", ws]].copy()
            df_station = df_station.merge(to_merge, on="Unix Date", how="left")

    return df_station


###############################################
# Export individual prepared datasets
###############################################

# Make sure the output directory exists
output_dir = "Data_debit/clean_data_individual"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to keep all prepared dataframes, if you want to access them programmatically
prepared_datasets = {}

for idx, row in data_water_station_period_summary.iterrows():
    main_station = row["Water station"]

    # 1) Build the base dataframe for this station (no features yet)
    df_prepared = prepare_single_station(
        row,
        data_precipitation_clean_merge,
        data_debit_clean_merge
    )

    # 2) Determine water station columns in this dataframe
    #    (all station columns that are not time and not weather)
    water_station_cols = [
        c for c in df_prepared.columns
        if c not in ["ISO Date", "Unix Date"] and not c.startswith("W_")
    ]

    # 3) Add rolling sums for weather stations (precipitation)
    df_prepared = add_weather_rolling_precip(df_prepared, windows_days=(14, 30))

    # 4) Add rolling means for water stations (debit)
    df_prepared = add_water_rolling_debit(
        df_prepared,
        water_station_columns=water_station_cols,
        windows_days=(14, 30)
    )

    # 5) Add time features (week of the year)
    df_prepared = add_time_features(df_prepared)

    # Store in dict
    prepared_datasets[main_station] = df_prepared

    # Optional: create a valid name for variable & file
    safe_name = re.sub(r"\W+", "_", str(main_station))

    # Create a variable like station_name_prepared (global scope)
    globals()[f"{safe_name}_prepared"] = df_prepared

    # Save to CSV: Data_debit/clean_data_individual/<station_name>_prepared.csv
    output_path = os.path.join(output_dir, f"{safe_name}_prepared.csv")
    df_prepared.to_csv(output_path, index=False)

    print(f"Prepared data for station '{main_station}' saved to '{output_path}'")