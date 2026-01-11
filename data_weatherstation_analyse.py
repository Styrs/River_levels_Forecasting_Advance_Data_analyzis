import pandas as pd
import numpy as np
import data_clearing_function



data_precipitation_merge_raw = pd.read_csv("Data_meteo/data_precipitation_merge_raw.csv", 
                                   encoding="latin1",sep=",", na_values=["NAN", "NaN", "nan", "NA", ""])




#######################################################
#Checking the data
#######################################################



#------------------------------------------------------
#Checking there is duplicate in one station mesurement
#------------------------------------------------------
def count_duplicates_for_station(df, station):
        temp = df[['Unix Date', station]]
        duplicates = temp[temp.duplicated(subset=['Unix Date'], keep=False)]
        return duplicates



stations_name = data_precipitation_merge_raw.columns.drop(['ISO Date', 'Unix Date'])
duplicate_summary = {}

for station in stations_name:
    dups = count_duplicates_for_station(data_precipitation_merge_raw, station)
    duplicate_summary[station] = len(dups)

duplicate_summary

## Warning for duplicates ##

has_duplicates = False

for station, count in duplicate_summary.items():
    if count > 0:
        print(f" WARNING: Station '{station}' has {count} duplicated timestamps.")
        has_duplicates = True

if not has_duplicates:
    print(" No time duplicates detected for any station.")


#------------------------------------------------------
#Checking the NA count:
#------------------------------------------------------

station_name = data_precipitation_merge_raw.columns.drop(['ISO Date', 'Unix Date'])

na_summary = data_clearing_function.count_na_data_column(data_precipitation_merge_raw,station_name)


for station, stats in na_summary.items():
    print(f"Station {station}: rows={stats['total_rows']}, NA={stats['na_count']}, NA%={stats['na_percent']:.2f}%")
    if stats['na_percent'] > 79:
        print(f"Station {station} have more than 80% NA ! ({stats['na_percent']})")



#------------------------------------------------------
#Checking the NA distribution:
#------------------------------------------------------


stations_name = data_precipitation_merge_raw.columns.drop(['ISO Date', 'Unix Date'])
time_column_computation = 'Unix Date'
time_column_display = 'ISO Date'
longest_periods = data_clearing_function.longest_non_na_period(data_precipitation_merge_raw,stations_name,time_column_computation,time_column_display)

for station, info in longest_periods.items():
    if info is None:
        continue
    print(
        f"Station {station}: longest continuous non-NA period from "
        f"{info['start_display']} to {info['end_display']} "
        f"(~{info['duration_days']:.2f} days)"
    )

#------------------------------------------------------
# NA gap length distribution per station
#------------------------------------------------------

stations_name = data_precipitation_merge_raw.columns.drop(['ISO Date', 'Unix Date'])

na_gap_stats = data_clearing_function.na_gap_length_distribution (data_precipitation_merge_raw,stations_name)


# Pretty print
for station, counts in na_gap_stats.items():
    print(f"\nStation {station}:")
    for bin_label, value in counts.items():
        print(f"  NA run length {bin_label}: {value} occurrences")
print("=========================== END OF FIRST ANALYSE ===========================")


#######################################################
# Transform the data
#######################################################

def clean_precipitation_data(
    data_precipitation_merge_raw,
    NA_period_to_smooth=24
):
    """
    Clean and prepare weather station data.

    Parameters
    ----------
    data_precipitation_merge_raw : pd.DataFrame
        Raw merged precipitation data. Must contain 'ISO Date' and 'Unix Date'.
    NA_period_to_smooth : int
        Maximum length of NA gaps to fill (default = 24).

    Returns
    -------
    data_precipitation_clean_merge : pd.DataFrame
        Fully cleaned precipitation dataset.
    """

    #------------------------------------------------------
    # 1. Force numerical conversion
    #------------------------------------------------------
    stations_name = data_precipitation_merge_raw.columns.drop(['ISO Date', 'Unix Date'])
    data_precipitation_clean_merge, conversion_report = data_clearing_function.force_numeric_columns(
        data_precipitation_merge_raw,
        stations_name
    )

    #------------------------------------------------------
    # 2. Remove first & last NA periods
    #------------------------------------------------------
    stations_name = data_precipitation_clean_merge.columns.drop(['ISO Date', 'Unix Date'])
    data_precipitation_clean_merge = data_clearing_function.GetRid_firstNdLast_NA(
        data_precipitation_clean_merge,
        stations_name
    )

    #------------------------------------------------------
    # 3. Compute NA summary and remove stations with >79% NA
    #------------------------------------------------------
    na_summary = data_clearing_function.count_na_data_column(
        data_precipitation_clean_merge,
        stations_name
    )

    column_to_be_deleted = [
        station for station, stats in na_summary.items()
        if stats['na_percent'] > 79
    ]

    if column_to_be_deleted:
        data_precipitation_clean_merge = data_clearing_function.remove_columns(
            data_precipitation_clean_merge,
            column_to_be_deleted
        )

    #------------------------------------------------------
    # 4. Fill small NA gaps
    #------------------------------------------------------
    stations_name = data_precipitation_clean_merge.columns.drop(['ISO Date', 'Unix Date'])
    data_precipitation_clean_merge, _, _ = data_clearing_function.fill_small_na_gaps_with_avg(
        data_precipitation_clean_merge,
        stations_name,
        NA_period_to_smooth
    )

    return data_precipitation_clean_merge

#######################################################
#Running new data check up
#######################################################



data_precipitation_clean_merge = clean_precipitation_data(data_precipitation_merge_raw,NA_period_to_smooth=24)
data_precipitation_clean_merge.to_csv("Data_meteo/data_precipitation_clean_merge.csv", index=False, encoding="utf-8")


stations_name = data_precipitation_clean_merge.columns.drop(['ISO Date', 'Unix Date'])

na_counts = data_clearing_function.count_na_data_column(data_precipitation_clean_merge,stations_name)

longest_periods = data_clearing_function.longest_non_na_period(data_precipitation_clean_merge,stations_name,'Unix Date','ISO Date')

na_gap_stats =data_clearing_function.na_gap_length_distribution(data_precipitation_clean_merge,stations_name)



for station in stations_name:
    print(f"===== For {station} ======")

    # Basic NA stats
    print(f"row number={na_counts[station]['total_rows']}, "
          f"NA number={na_counts[station]['na_count']}, "
          f"NA%={na_counts[station]['na_percent']:.2f}%")

    # Longest non-NA period
    lp = longest_periods[station]
    print(
        f"Station {station}: longest continuous non-NA period from "
        f"{lp['start_display']} to {lp['end_display']} "
        f"(~{lp['duration_days']:.2f} days)"
    )

    # NA gap length distribution *for this station only*
    print("NA gap length distribution:")
    for bin_label, value in na_gap_stats[station].items():
        if value > 0:        # ← only print non-zero bins
            print(f"  {bin_label}: {value} occurrences")

    print()  # blank line between stations

#print("let's go !")
#overlap_info = data_clearing_function.best_overlap_by_k(longest_periods)
#print("It worked ?")



#for k, info in sorted(overlap_info.items()):
#    print(f"\n=== Longest period with at least {k} stations ===")
#    print(f"Stations: {info['stations']}")
#    print(f"From {info['overlap_start']} to {info['overlap_end']}")
#    print(f"Duration: ~{info['overlap_days']:.2f} days")