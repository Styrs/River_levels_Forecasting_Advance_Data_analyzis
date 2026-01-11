import pandas as pd
import numpy as np
import data_clearing_function
import data_weatherstation_analyse

# Load your CSV
Data_debit_raw_merge = pd.read_csv("Data_debit/Data_debit_raw_merge.csv", 
                                   encoding="latin1",sep=",", na_values=["NAN", "NaN", "nan", "NA", ""])


print(Data_debit_raw_merge.columns)


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



stations_name = Data_debit_raw_merge.columns.drop(['ISO Date', 'Unix Date'])
duplicate_summary = {}

for station in stations_name:
    dups = count_duplicates_for_station(Data_debit_raw_merge, station)
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

stations_name = Data_debit_raw_merge.columns.drop(['ISO Date', 'Unix Date'])

na_summary = data_clearing_function.count_na_data_column(Data_debit_raw_merge,stations_name)



for station, stats in na_summary.items():
    print(f"Station {station}: rows={stats['total_rows']}, NA={stats['na_count']}, NA%={stats['na_percent']:.2f}%")
    if stats['na_percent'] > 79:
        print(f"Station {station} have more than 80% NA ! ({stats['na_percent']})")



#------------------------------------------------------
#Checking the NA distribution:
#------------------------------------------------------


stations_name = Data_debit_raw_merge.columns.drop(['ISO Date', 'Unix Date'])
time_column_computation = 'Unix Date'
time_column_display = 'ISO Date'
longest_periods = data_clearing_function.longest_non_na_period(Data_debit_raw_merge,stations_name,time_column_computation,time_column_display)


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

stations_name = Data_debit_raw_merge.columns.drop(['ISO Date', 'Unix Date'])

na_gap_stats = data_clearing_function.na_gap_length_distribution (Data_debit_raw_merge,stations_name)



# Pretty print
for station, counts in na_gap_stats.items():
    print(f"\nStation {station}:")
    for bin_label, value in counts.items():
        print(f"  NA run length {bin_label}: {value} occurrences")
print("=========================== END OF FIRST ANALYSE ===========================")




def clean_debit_data(Data_debit_raw_merge, na_summary, data_clearing_function,
                     na_threshold=79, max_gap_hours=24):
    """
    Clean debit station data:
      1. Convert station columns to numeric.
      2. Remove leading/trailing NA-only periods.
      3. Remove stations with NA percentage above `na_threshold`.
      4. Fill small NA gaps (up to `max_gap_hours`).

    Returns
    -------
    data_debit_clean_merge : pd.DataFrame
        Fully cleaned dataset.
    """

    # 1. Force station columns to numeric
    stations_name = Data_debit_raw_merge.columns.drop(['ISO Date', 'Unix Date'])
    data_debit_clean_merge, _ = data_clearing_function.force_numeric_columns(
        Data_debit_raw_merge.copy(),
        stations_name
    )

    # 2. Remove first/last NA periods
    stations_name = data_debit_clean_merge.columns.drop(['ISO Date', 'Unix Date'])
    data_debit_clean_merge = data_clearing_function.GetRid_firstNdLast_NA(
        data_debit_clean_merge,
        stations_name
    )

    # 3. Remove stations with too many NAs
    columns_to_delete = [
        station for station, stats in na_summary.items()
        if stats['na_percent'] > na_threshold
    ]

    if columns_to_delete:
        data_debit_clean_merge = data_clearing_function.remove_columns(
            data_debit_clean_merge,
            columns_to_delete
        )

    # 4. Fill small NA gaps
    stations_name = data_debit_clean_merge.columns.drop(['ISO Date', 'Unix Date'])
    data_debit_clean_merge, _, _ = data_clearing_function.fill_small_na_gaps_with_avg(
        data_debit_clean_merge,
        stations_name,
        max_gap_hours
    )

    return data_debit_clean_merge

#------------------------------------------------------
#Running new data check up
#------------------------------------------------------
data_debit_clean_merge = clean_debit_data(Data_debit_raw_merge, na_summary, data_clearing_function)
data_debit_clean_merge.to_csv("Data_debit/data_debit_clean_merge.csv", index=False, encoding="utf-8")

stations_name = data_debit_clean_merge.columns.drop(['ISO Date', 'Unix Date'])

na_counts = data_clearing_function.count_na_data_column(data_debit_clean_merge,stations_name)

longest_periods = data_clearing_function.longest_non_na_period(data_debit_clean_merge,stations_name,'Unix Date','ISO Date')

na_gap_stats =data_clearing_function.na_gap_length_distribution(data_debit_clean_merge,stations_name)



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




#################################################################################
# Let's select the the weatherstation data that match the debit data period #
#################################################################################

#have to run the weatherstation analyse first to have the longest period info
data_precipitation_merge_raw = pd.read_csv("Data_meteo/data_precipitation_merge_raw.csv", 
                                   encoding="latin1",sep=",", na_values=["NAN", "NaN", "nan", "NA", ""])

data_precipitation_clean_merge = data_weatherstation_analyse.clean_precipitation_data(data_precipitation_merge_raw,NA_period_to_smooth=24)




weatherstation_period = data_clearing_function.longest_non_na_period(data_precipitation_clean_merge,
                                                                   data_precipitation_clean_merge.columns.drop(['ISO Date', 'Unix Date']),'Unix Date','ISO Date')
water_to_weather = data_clearing_function.match_water_to_weather(longest_periods, weatherstation_period)

print(longest_periods)
print("=== WATER → WEATHER STATION MATCHES ===\n")

for water_station, weather_list in water_to_weather.items():
    print(f"Water station: {water_station}")

    if len(weather_list) == 0:
            print("  → No matching weather stations found.\n")
    else:
        print("  → Matching weather stations:")
        
        print(f"      - {weather_list}")
        print()  

#------------------------------------------------------
#Creating a summary table of the water station periods with matching weather stations and other water station 
#------------------------------------------------------


# --- your existing code ---

water_station_period_summary = {}

for water_station, info in longest_periods.items():
    water_station_period_summary[water_station] = {
        'start_display': info['start_display'],
        'end_display':   info['end_display'],
        'start_time':    info['start_time'],
        'end_time':      info['end_time'],
        'duration':      info['duration'],
        'duration_days': info['duration_days'],
        'matching_weather_stations': water_to_weather.get(water_station, [])
    }

# Convert the dictionary to a DataFrame
Station_picking_summary = pd.DataFrame.from_dict(water_station_period_summary, orient='index')

# Convert the list of matching weather stations into a comma-separated string
Station_picking_summary['matching_weather_stations'] = Station_picking_summary['matching_weather_stations'].apply(
    lambda x: ", ".join(x) if isinstance(x, list) else ""
)

# --- NEW PART: use match_water_to_water and add column ---

# 1) Run the new function (defined earlier)
water_to_water = data_clearing_function.match_water_to_water(longest_periods)

# 2) Add a column with matching water stations (as comma-separated string)
Station_picking_summary['matching_water_stations'] = (
    Station_picking_summary.index.to_series()
    .apply(lambda ws: ", ".join(water_to_water.get(ws, [])))
)


Station_picking_summary.index.name = "Water station"

# --- Save to CSV (now includes both weather + water matches) ---

Station_picking_summary.to_csv("Data_debit/water_station_period_summary.csv",
                               encoding="utf-8",
                               index=True)