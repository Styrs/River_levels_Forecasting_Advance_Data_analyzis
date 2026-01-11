import pandas as pd
import requests
import os
import glob


list_temporal_period = ["2020-2029","2010-2019","2000-2009"]
list_weather_station = ['GVE','CGI','DOL','LON','BIE','LSN','PUY','COS','CHB','VIT','VEV','ORO','MAH','AUB','FRE',
                        'COU','PAY','GRA','MAS','AVA','BOU','AIG','CDM','BEX','VSCHO','EVI','MAB']

output_folder = "C:/Unil/Master/Automne 2/Ada/Projet/Data_meteo/raw_data_individuel"
os.makedirs(output_folder, exist_ok=True)



patterns = [
    {"folder": "ch.meteoschweiz.ogd-smn",
     "prefix": "ogd-smn"},
    {"folder": "ch.meteoschweiz.ogd-smn-precip",
     "prefix": "ogd-smn-precip"}]



for station in list_weather_station:
    station_code = station.lower()
    period_count = 1

    for period in list_temporal_period:
        filename = f"{station}_{period_count}.csv"
        filepath = os.path.join(output_folder, filename)

        success = False

        for pat in patterns:
            url = (
                f"https://data.geo.admin.ch/{pat['folder']}/"
                f"{station_code}/{pat['prefix']}_{station_code}_h_historical_{period}.csv"
            )

            response = requests.get(url)

            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"✔ Downloaded {filename} {period} using pattern {pat['prefix']}")
                success = True
                break  # stop trying other patterns for this station/period
            else:
                print(f"    ✖ Failed {filename} {period} with {pat['prefix']} (status {response.status_code})")

        if not success:
            print(f"    ⚠ No file found for {filename} {period} with any pattern")

        period_count += 1



### merge the files of the same station together 
folder = r"C:/Unil/Master/Automne 2/Ada/Projet/Data_meteo/raw_data_individuel"
output_folder = r"C:/Unil/Master/Automne 2/Ada/Projet/Data_meteo/merged"
os.makedirs(output_folder, exist_ok=True)

stations = set(f.split("_")[0] for f in os.listdir(folder) if f.endswith(".csv")) # create a list of the same station's files. 

for station in stations:
    pattern = os.path.join(folder, f"{station}_*.csv")
    files = sorted(glob.glob(pattern))

    dfs = []
    for file in files:
        # ★ IMPORTANT FIX ★
        df = pd.read_csv(file, sep=";")

        # Clean column names
        df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    merged["reference_timestamp"] = pd.to_datetime(
        merged["reference_timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    merged = merged.sort_values("reference_timestamp")

    merged.to_csv(
        os.path.join(output_folder, f"{station}.csv"),
        index=False,
        sep=";"
    )

    print(f"✔ Merged {station}")