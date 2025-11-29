import pandas as pd
import numpy as np
from pathlib import Path


def month_to_season(month: int) -> int:
    if month in (11, 12, 1, 2):
        return 1
    if month in (3, 4):
        return 2
    if month in (5, 6):
        return 3
    return 4


def load_global_pollution(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "country_name": "Country",
        "city_name": "City",
        "aqi_value": "AQI",
        "pm2.5_aqi_value": "PM2.5",
        "no2_aqi_value": "NO2",
        "ozone_aqi_value": "Ozone",
        "co_aqi_value": "CO",
    })
    df["Year"] = 2019
    df["State"] = df.get("City")
    keep_cols = ["Country", "State", "Year", "AQI", "PM2.5", "NO2", "Ozone", "CO"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep_cols]


def load_cause_of_deaths(path: str, population_map: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Year"] == 2019].copy()
    df = df.rename(columns={
        "Country/Territory": "Country",
        "Cardiovascular Diseases": "Cardio",
        "Lower Respiratory Infections": "Lower_Resp",
        "Chronic Respiratory Diseases": "Chronic_Resp",
    })

    def per_100k(row, col):
        pop = population_map.get(row["Country"])
        return (row[col] / pop) * 1e5 if pop else np.nan

    df["Cardiovascular_deaths_per_100k"] = df.apply(lambda r: per_100k(r, "Cardio"), axis=1)
    df["Respiratory_deaths_per_100k"] = df.apply(
        lambda r: per_100k(r, "Lower_Resp") + per_100k(r, "Chronic_Resp"), axis=1
    )
    df["Combined_disease_risk_score"] = df["Cardiovascular_deaths_per_100k"] + df["Respiratory_deaths_per_100k"]

    return df[["Country", "Year", "Cardiovascular_deaths_per_100k", "Respiratory_deaths_per_100k", "Combined_disease_risk_score"]]


def load_india_city_day(path: str) -> pd.DataFrame:
    required = {"City", "Date", "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NOx"}
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in city_day: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Year"] = df["Date"].dt.year
    df = df[df["Year"].between(2015, 2019)]
    df["State"] = df.get("State", df["City"])
    return df


def aggregate_india(df: pd.DataFrame) -> pd.DataFrame:
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NOx"]
    agg_dict = {col: "mean" for col in pollutants}
    agg_dict.update({
        "AQI": ["mean", "max", "std"],
    })
    grouped = df.groupby(["State", "Year"]).agg(agg_dict).reset_index()
    grouped.columns = [
        "State",
        "Year",
        *pollutants,
        "AQI",
        "AQI_max",
        "AQI_std",
    ]
    grouped["Country"] = "India"
    grouped = grouped[["Country", "State", "Year", "AQI", *pollutants, "AQI_max", "AQI_std"]]
    return grouped


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Pairwise
    df["PM25_NO2"] = df["PM2.5"] * df["NO2"]
    df["PM25_Ozone"] = df["PM2.5"] * df.get("Ozone", np.nan)
    df["PM25_SO2"] = df["PM2.5"] * df.get("SO2", np.nan)
    df["NO2_SO2"] = df["NO2"] * df.get("SO2", np.nan)
    df["NO2_Ozone"] = df["NO2"] * df.get("Ozone", np.nan)
    df["PM25_NO2_SO2"] = df["PM2.5"] * df["NO2"] * df.get("SO2", np.nan)

    # Polynomial
    df["PM25_squared"] = df["PM2.5"] ** 2
    df["PM25_cubed"] = df["PM2.5"] ** 3
    df["NO2_squared"] = df["NO2"] ** 2
    df["NO2_cubed"] = df["NO2"] ** 3
    if "AQI" in df.columns:
        df["AQI_squared"] = df["AQI"] ** 2

    # Ratios
    df["PM25_over_NO2"] = df["PM2.5"] / df["NO2"].replace(0, np.nan)
    if "PM10" in df.columns:
        df["PM10_over_PM25"] = df["PM10"] / df["PM2.5"].replace(0, np.nan)
    if "NOx" in df.columns:
        df["NOx_over_NO2"] = df["NOx"] / df["NO2"].replace(0, np.nan)

    return df


def prepare_model4_data(
    city_path: str = "city_day.csv",
    global_pollution_path: str = "global_air_pollution_data.csv",
    deaths_path: str = "cause_of_deaths.csv",
    output_path: str = "model4_pollutant_synergy.csv",
) -> Path:
    # Minimal population map to compute per-100k (only India known here)
    population_map = {
        "India": 1_368_000_000,  # 2019 approx
    }

    global_poll = load_global_pollution(global_pollution_path)
    deaths = load_cause_of_deaths(deaths_path, population_map)

    global_df = global_poll.merge(deaths, on=["Country", "Year"], how="left")
    global_df["is_india"] = (global_df["Country"].str.lower() == "india").astype(int)

    india_city = load_india_city_day(city_path)
    india_agg = aggregate_india(india_city)
    india_deaths = deaths[deaths["Country"].str.lower() == "india"].copy()
    india_agg = india_agg.merge(india_deaths.drop(columns=["Country"]), on="Year", how="left")
    india_agg["is_india"] = 1

    combined = pd.concat([global_df, india_agg], ignore_index=True, sort=False)
    combined = add_interaction_features(combined)

    # Order columns
    ordered_cols = [
        "Country",
        "State",
        "Year",
        "is_india",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "Ozone",
        "NOx",
        "AQI",
        "AQI_max",
        "AQI_std",
        "Cardiovascular_deaths_per_100k",
        "Respiratory_deaths_per_100k",
        "Combined_disease_risk_score",
        # interactions
        "PM25_NO2",
        "PM25_Ozone",
        "PM25_SO2",
        "NO2_SO2",
        "NO2_Ozone",
        "PM25_NO2_SO2",
        # polynomial
        "PM25_squared",
        "PM25_cubed",
        "NO2_squared",
        "NO2_cubed",
        "AQI_squared",
        # ratios
        "PM25_over_NO2",
        "PM10_over_PM25",
        "NOx_over_NO2",
    ]

    for col in ordered_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    combined = combined[ordered_cols]

    # Impute numeric columns with median
    num_cols = combined.select_dtypes(include=[np.number]).columns
    medians = combined[num_cols].median()
    combined.loc[:, num_cols] = combined[num_cols].fillna(medians)

    combined.to_csv(output_path, index=False)
    print(f"Saved {output_path} with {len(combined)} rows and {len(combined.columns)} columns.")
    print("Columns:", combined.columns.tolist())
    return Path(output_path)


if __name__ == "__main__":
    prepare_model4_data()
