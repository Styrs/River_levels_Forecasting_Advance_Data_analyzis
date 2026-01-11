import pandas as pd
import numpy as np
from itertools import combinations



def count_na_data_column(data_frame, column_list):
    """
    Returns basic stats for each column in column_list:
    - total number of rows
    - number of NA values
    - percentage of NAs
    """

    
    results = {}

    for col in column_list:

        if col not in data_frame.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        total_rows = len(data_frame[col])
        na_count = data_frame[col].isna().sum()
        na_percent = (na_count / total_rows) * 100

        # store results for this column
        results[col] = {
            "total_rows": total_rows,
            "na_count": na_count,
            "na_percent": na_percent
        }

    return results



def longest_non_na_period(data_frame,station_list,time_col,display_time_col=None):
    """
    For each station column in station_list, finds the longest continuous period
    of non-NA values based on a given time column.
    """

    # Default display column = time_col
    if display_time_col is None:
        display_time_col = time_col

    # Basic checks
    if time_col not in data_frame.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

    if display_time_col not in data_frame.columns:
        raise ValueError(f"Display time column '{display_time_col}' not found in DataFrame.")

    longest_periods = {}

    for station in station_list:

        if station not in data_frame.columns:
            raise ValueError(f"Column '{station}' not found in DataFrame.")

        station_data = data_frame[[display_time_col, time_col, station]].copy()
        mask = station_data[station].notna()

        if not mask.any():
            longest_periods[station] = None
            continue

        group_id = (mask != mask.shift(fill_value=False)).cumsum()

        sub_non_na = station_data[mask].copy()
        group_id_non_na = group_id[mask]

        best_info = None
        best_duration = -1

        for gid, g in sub_non_na.groupby(group_id_non_na):
            start_time = g[time_col].iloc[0]
            end_time   = g[time_col].iloc[-1]
            duration   = end_time - start_time

            if duration > best_duration:
                best_duration = duration
                best_info = {
                    'start_display': g[display_time_col].iloc[0],
                    'end_display':   g[display_time_col].iloc[-1],
                    'start_time':    start_time,
                    'end_time':      end_time,
                    'duration':      duration,
                    'duration_days': duration / (24 * 3600)
                }

        longest_periods[station] = best_info

    return longest_periods



def na_gap_length_distribution(
    data_frame,
    column_list,
    bins=None
):
    """
    Compute the distribution of consecutive NA run lengths for each column.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        The full dataset.

    column_list : list of str
        List of column names (e.g. station names) for which to compute NA gaps.

    bins : list of tuples or None, optional
        Binning scheme for NA run lengths.
        Each element is (label, min_len, max_len) where:
          - min_len and max_len are inclusive
          - if max_len is None, it means "infinity"

        If None, the following default is used:
            [("1", 1, 1),
             ("2", 2, 2),
             ("3-5", 3, 5),
             ("6-10", 6, 10),
             ("11-20", 11, 20),
             ("21-50", 21, 50),
             ("51-100", 51, 100),
             (">100", 101, None)]

    Returns
    -------
    dict
        {
          column_name: {
              bin_label: count,
              ...
          },
          ...
        }
    """



    # Default bins (you can adapt labels / ranges)
    if bins is None:
        bins = [
            ("1", 1, 1),
            ("2", 2, 2),
            ("3-5", 3, 5),
            ("6-10", 6, 10),
            ("11-20", 11, 20),
            ("21-50", 21, 50),
            ("51-100", 51, 100),
            (">100", 101, None),  # open-ended
        ]

    # Prepare output dict
    na_gap_stats = {}

    # Pre-build zero counts template for a station with no NAs
    empty_counts_template = {label: 0 for (label, _, _) in bins}

    for col in column_list:

        if col not in data_frame.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Work only with the column itself (no need for dates here)
        series = data_frame[col]

        # True where we have NA, False where we have data
        mask_na = series.isna()

        # If no NA at all: just store zeros for all bins
        if not mask_na.any():
            na_gap_stats[col] = empty_counts_template.copy()
            continue

        # Identify consecutive runs of NA (True in mask_na)
        group_id = (mask_na != mask_na.shift(fill_value=False)).cumsum()

        # Keep only rows where we have NA
        group_id_na = group_id[mask_na]
        sub_na = series[mask_na]

        # Initialize counts for this column
        counts = empty_counts_template.copy()

        # For each continuous NA run, compute its length and bin it
        for _, g in sub_na.groupby(group_id_na):
            run_len = len(g)

            # Find which bin this length belongs to
            for label, min_len, max_len in bins:
                if run_len < min_len:
                    continue
                if max_len is not None and run_len > max_len:
                    continue
                counts[label] += 1
                break  # stop after first matching bin

        na_gap_stats[col] = counts

    return na_gap_stats



def GetRid_firstNdLast_NA(data_frame, checking_columns):
    """
    Trim rows at the beginning and end of a DataFrame where *all* of the
    checking_columns are NA. Middle rows are never removed, even if they are all NA.
    """

    # 1. Make sure all requested columns exist
    missing = [c for c in checking_columns if c not in data_frame.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # 2. True where ALL checking columns are NA (row-wise)
    all_na = data_frame[checking_columns].isna().all(axis=1)

    # 3. True where NOT all checking columns are NA (so: at least one non-NA)
    non_all_na = ~all_na

    # 4. Edge case: if there is no useful data at all
    if not non_all_na.any():
        return data_frame.iloc[0:0].copy()

    # 5. Find first and last row that has at least one non-NA value
    first_index = non_all_na.idxmax()
    last_index = non_all_na[::-1].idxmax()

    # 6. Return the slice between them (inclusive)
    return data_frame.loc[first_index:last_index].copy()



def remove_columns(data_frame, column_list, inplace=False):
    """
    Remove the given columns from a DataFrame.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        The original DataFrame.

    column_list : list of str
        List of column names to remove.

    inplace : bool, optional (default False)
        If True, modify the original DataFrame.
        If False, return a new DataFrame with the columns removed.

    Returns
    -------
    pandas.DataFrame
        The DataFrame without the specified columns (unless inplace=True).
    """

    # Check for missing columns
    missing = [col for col in column_list if col not in data_frame.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    if inplace:
        data_frame.drop(columns=column_list, inplace=True)
        return data_frame
    else:
        return data_frame.drop(columns=column_list).copy()
    


def fill_small_na_gaps_with_avg(data_frame, column_list, max_gap):

    if max_gap < 1:
        raise ValueError("max_gap must be at least 1.")

    missing = [c for c in column_list if c not in data_frame.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    df = data_frame.copy()

    # Track NA BEFORE filling
    na_before = {col: df[col].isna().sum() for col in column_list}

    for col in column_list:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        s = df[col]
        is_na = s.isna()

        if not is_na.any():
            continue

        group_id = (is_na != is_na.shift(fill_value=False)).cumsum()
        na_groups = group_id[is_na]

        for _, g in s[is_na].groupby(na_groups):
            run_len = len(g)
            if run_len > max_gap:
                continue

            first_idx = g.index[0]
            last_idx = g.index[-1]

            first_pos = s.index.get_loc(first_idx)
            last_pos = s.index.get_loc(last_idx)

            if first_pos == 0 or last_pos == len(s) - 1:
                continue

            prev_val = s.iloc[first_pos - 1]
            next_val = s.iloc[last_pos + 1]

            if pd.isna(prev_val) or pd.isna(next_val):
                continue

            fill_value = (prev_val + next_val) / 2.0
            df.loc[g.index, col] = fill_value

    # NA AFTER filling
    na_after = {col: df[col].isna().sum() for col in column_list}

    # NA filled = before - after
    na_filled = {col: na_before[col] - na_after[col] for col in column_list}

    return df, na_after, na_filled



def force_numeric_columns(data_frame, column_list):

    """
    Ensures that all columns in column_list are numeric.
    Converts values to numerical types and leaves valid NA values untouched.
    Non-numeric values are turned into NaN.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        The input DataFrame.

    column_list : list of str
        Columns to convert to numeric.

    Returns
    -------
    (pandas.DataFrame, dict)
        - A new DataFrame with numeric columns.
        - A dict reporting how many values became NaN in each column.
    """

    # Check existence of columns
    missing = [c for c in column_list if c not in data_frame.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    df = data_frame.copy()
    conversion_report = {}

    for col in column_list:
        # Count NA before conversion
        before_na = df[col].isna().sum()

        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Count NA after conversion
        after_na = df[col].isna().sum()

        # How many values became NA because they were not numeric?
        new_na = after_na - before_na

        conversion_report[col] = {
            "na_before": before_na,
            "na_after": after_na,
            "new_na_created": new_na
        }
    
    return df, conversion_report


def _common_overlap(stations, longest_periods):
    starts = [longest_periods[s]['start_time'] for s in stations]
    ends   = [longest_periods[s]['end_time']   for s in stations]

    overlap_start = max(starts)
    overlap_end   = min(ends)
    overlap_duration = overlap_end - overlap_start

    if overlap_duration <= 0:
        return None, None, 0

    return overlap_start, overlap_end, overlap_duration


def best_overlap_by_k(longest_periods):
    stations = list(longest_periods.keys())
    n = len(stations)
    result = {}

    for k in range(1, n + 1):

        # 👇 NEW LINE: show progress
        print(f"Computing best combination for k = {k} stations...")

        best = {
            'stations': None,
            'overlap_start': None,
            'overlap_end': None,
            'overlap_duration': -1,
            'overlap_days': -1,
        }

        for combo in combinations(stations, k):
            if k == 1:
                s = combo[0]
                overlap_start = longest_periods[s]['start_time']
                overlap_end   = longest_periods[s]['end_time']
                overlap_duration = overlap_end - overlap_start
            else:
                overlap_start, overlap_end, overlap_duration = _common_overlap(combo, longest_periods)

            if overlap_duration > best['overlap_duration']:
                best['stations'] = combo
                best['overlap_start'] = overlap_start
                best['overlap_end'] = overlap_end
                best['overlap_duration'] = overlap_duration
                best['overlap_days'] = overlap_duration / 86400 if overlap_duration > 0 else 0

        result[k] = best

    return result


def match_water_to_weather(longest_periods, weatherstation_period):
    """
    For each water station, find the weather stations whose longest non-NA period
    fully covers the water station's longest non-NA period.

    Parameters
    ----------
    longest_periods : dict
        Dict for water stations, e.g.
        {
            'WaterStation_1': {
                'start_time': ...,
                'end_time': ...,
                ...
            },
            ...
        }

    weatherstation_period : dict
        Dict for weather stations with same structure, e.g.
        {
            'WeatherStation_A': {
                'start_time': ...,
                'end_time': ...,
                ...
            },
            ...
        }

    Returns
    -------
    dict
        Mapping:
        {
            'WaterStation_1': ['WeatherStation_A', 'WeatherStation_C', ...],
            'WaterStation_2': [...],
            ...
        }
    """

    water_to_weather = {}

    for water_station, w_info in longest_periods.items():
        ws_start = w_info['start_time']
        ws_end   = w_info['end_time']

        matching_weather_stations = []

        for weather_station, wx_info in weatherstation_period.items():
            wx_start = wx_info['start_time']
            wx_end   = wx_info['end_time']

            # strict coverage: weather station period fully contains water station period
            if wx_start <= ws_start and wx_end >= ws_end:
                matching_weather_stations.append(weather_station)

        water_to_weather[water_station] = matching_weather_stations

    return water_to_weather


def match_water_to_water(longest_periods):
    """
    For each water station, find the *other* water stations whose longest non-NA
    period fully covers that station's longest non-NA period.

    Parameters
    ----------
    longest_periods : dict
        Dict for water stations, e.g.
        {
            'WaterStation_1': {
                'start_time': ...,
                'end_time': ...,
                ...
            },
            ...
        }

    Returns
    -------
    dict
        Mapping:
        {
            'WaterStation_1': ['WaterStation_2', 'WaterStation_5', ...],
            'WaterStation_2': [...],
            ...
        }
    """

    water_to_water = {}

    for ws_name, ws_info in longest_periods.items():
        ws_start = ws_info['start_time']
        ws_end   = ws_info['end_time']

        matching_stations = []

        for other_name, other_info in longest_periods.items():
            # skip self
            if other_name == ws_name:
                continue

            o_start = other_info['start_time']
            o_end   = other_info['end_time']

            # strict coverage: other station period fully contains this station period
            if o_start <= ws_start and o_end >= ws_end:
                matching_stations.append(other_name)

        water_to_water[ws_name] = matching_stations

    return water_to_water


def add_column_from_dataset(object_data, import_data, column_name):
    """
    Add a column from import_data into object_data.
    
    Parameters
    ----------
    object_data : pd.DataFrame
        The dataframe to which the column will be added.
    import_data : pd.DataFrame
        The dataframe from which the column will be taken.
    column_name : str
        Name of the column to copy.
    
    Returns
    -------
    new_dataset : pd.DataFrame
        A new dataframe with the added column.
    """

    # Create a copy to avoid modifying the original
    new_dataset = object_data.copy()

    # Add the column (pandas aligns automatically by index)
    new_dataset[column_name] = import_data[column_name]

    return new_dataset







