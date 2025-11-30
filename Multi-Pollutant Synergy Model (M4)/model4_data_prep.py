import pandas as pd
import numpy as np
from pathlib import Path


def load_pollution(global_path: str, city_path: str) -> pd.DataFrame:
    g = pd.read_csv(global_path)
    g.columns = [c.strip() for c in g.columns]
    g = g.rename(
        columns={
            "country_name": "Country",
            "city_name": "State",
            "aqi_value": "AQI",
            "pm2.5_aqi_value": "PM2.5",
            "no2_aqi_value": "NO2",
            "ozone_aqi_value": "Ozone",
            "co_aqi_value": "CO",
        }
    )
    g["Year"] = 2019
    # Aggregate to country-year
    g_country = g.groupby(["Country", "Year"]).agg(
        {
            "PM2.5": "mean",
            "NO2": "mean",
            "Ozone": "mean",
            "CO": "mean",
            "AQI": "mean",
        }
    ).reset_index()

    # Optional India state aggregates (2015-2019) for richer variation
    city = pd.read_csv(city_path)
    city["Date"] = pd.to_datetime(city["Date"], errors="coerce")
    city = city.dropna(subset=["Date"])
    city["Year"] = city["Date"].dt.year
    city = city[(city["Year"] >= 2015) & (city["Year"] <= 2019)]
    state_agg = (
        city.groupby(["City", "Year"]).agg(
            {
                "PM2.5": "mean",
                "PM10": "mean",
                "NO2": "mean",
                "SO2": "mean",
                "CO": "mean",
                "O3": "mean",
                "AQI": "mean",
            }
        )
    ).reset_index().rename(columns={"City": "State"})
    state_agg["Country"] = "India"

    return g_country, state_agg


def load_deaths(deaths_path: str) -> pd.DataFrame:
    d = pd.read_csv(deaths_path)
    d = d.rename(
        columns={
            "Country/Territory": "Country",
            "Cardiovascular Diseases": "Cardio",
            "Lower Respiratory Infections": "Lower_Resp",
            "Chronic Respiratory Diseases": "Chronic_Resp",
            "Neoplasms": "Neoplasms",
        }
    )
    d = d[d["Year"] == 2019].copy()
    d["Respiratory_deaths_per_100k"] = d["Lower_Resp"] + d["Chronic_Resp"]
    d["Cardiovascular_deaths_per_100k"] = d["Cardio"]
    d["Combined_disease_risk_score"] = d["Cardiovascular_deaths_per_100k"] + d["Respiratory_deaths_per_100k"] + d["Neoplasms"]
    return d[[
        "Country",
        "Year",
        "Cardiovascular_deaths_per_100k",
        "Respiratory_deaths_per_100k",
        "Combined_disease_risk_score",
    ]]


def build_dataset(global_poll_path: str, city_path: str, deaths_path: str, output_path: str) -> Path:
    g_country, india_state = load_pollution(global_poll_path, city_path)
    deaths = load_deaths(deaths_path)

    global_merged = g_country.merge(deaths, on=["Country", "Year"], how="inner")
    india_merged = india_state.merge(deaths[deaths["Country"] == "India"], on=["Country", "Year"], how="left")

    combined = pd.concat([global_merged, india_merged], ignore_index=True, sort=False)

    # Simple interactions from context correlations
    combined["PM25_NO2"] = combined.get("PM2.5", np.nan) * combined.get("NO2", np.nan)
    combined["PM25_SO2"] = combined.get("PM2.5", np.nan) * combined.get("SO2", np.nan)
    combined["PM25_CO"] = combined.get("PM2.5", np.nan) * combined.get("CO", np.nan)
    combined["NO2_SO2"] = combined.get("NO2", np.nan) * combined.get("SO2", np.nan)
    combined["SO2_CO"] = combined.get("SO2", np.nan) * combined.get("CO", np.nan)

    # Median impute numeric
    num_cols = combined.select_dtypes(include=[np.number]).columns
    medians = combined[num_cols].median()
    combined[num_cols] = combined[num_cols].fillna(medians)

    combined.to_csv(output_path, index=False)
    print(f"Saved {output_path} with {len(combined)} rows and {len(combined.columns)} columns.")
    return Path(output_path)


if __name__ == "__main__":
    build_dataset(
        global_poll_path="../global_air_pollution_data.csv",
        city_path="../city_day.csv",
        deaths_path="../cause_of_deaths.csv",
        output_path="model4_pollutant_synergy.csv",
    )
