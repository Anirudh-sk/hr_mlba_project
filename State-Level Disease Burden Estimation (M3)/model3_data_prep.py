import pandas as pd
import numpy as np
from pathlib import Path


def add_state_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a State column. Falls back to City when no mapping is available."""
    if "State" in df.columns:
        df["State"] = df["State"].fillna(df.get("City"))
    else:
        df["State"] = df["City"]
    return df


def compute_state_year_agg(city_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate city-day data to state-year with pollutant stats and AQI metrics."""
    city_df = add_state_column(city_df)
    city_df["Year"] = city_df["Date"].dt.year
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NOx"]

    agg_dict = {col: "mean" for col in pollutants}
    agg_dict.update({
        "AQI": ["mean", "max", "std"],
        "severe_flag": "mean",  # mean * 100 later
        "very_poor_flag": "mean",
        "Date": "count",  # day count
    })

    grouped = (
        city_df.groupby(["State", "Year"])
        .agg(agg_dict)
        .reset_index()
    )

    # Flatten columns
    grouped.columns = [
        "State",
        "Year",
        *pollutants,
        "mean_AQI",
        "max_AQI",
        "std_AQI",
        "pct_severe_days_raw",
        "pct_very_poor_days_raw",
        "num_days"
    ]

    grouped["pct_severe_days"] = grouped["pct_severe_days_raw"] * 100
    grouped["pct_very_poor_days"] = grouped["pct_very_poor_days_raw"] * 100
    grouped = grouped.drop(columns=["pct_severe_days_raw", "pct_very_poor_days_raw"])
    return grouped


def load_city_day(input_path: str) -> pd.DataFrame:
    required_cols = {"City", "Date", "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NOx"}
    csv_path = Path(input_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in city_day: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    df["severe_flag"] = (df["AQI"] >= 300).astype(int)
    df["very_poor_flag"] = (df["AQI"] >= 200).astype(int)
    df = df[(df["Date"].dt.year >= 2015) & (df["Date"].dt.year <= 2019)]
    return df


def load_global_pollution(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "country_name": "Country",
        "city_name": "City",
        "aqi_value": "AQI_value",
        "pm2.5_aqi_value": "PM2.5_value",
        "no2_aqi_value": "NO2_value",
        "ozone_aqi_value": "Ozone_value",
        "co_aqi_value": "CO_value",
    })
    df = df[df["Country"].str.lower() == "india"].copy()
    if df.empty:
        return pd.DataFrame(columns=["State", "PM2.5_value", "NO2_value", "Ozone_value", "AQI_value", "CO_value"])
    df = add_state_column(df)
    agg = df.groupby("State").agg({
        "PM2.5_value": "mean",
        "NO2_value": "mean",
        "Ozone_value": "mean",
        "AQI_value": "mean",
        "CO_value": "mean",
    }).reset_index()
    return agg


def load_disease_data(path: str, population_map: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Country/Territory"].str.lower() == "india"].copy()
    df = df[df["Year"].between(2015, 2019)]
    df = df.rename(columns={
        "Cardiovascular Diseases": "Cardiovascular",
        "Lower Respiratory Infections": "Lower_Respiratory",
        "Chronic Respiratory Diseases": "Chronic_Respiratory",
    })

    def per_100k(row, col):
        pop = population_map.get(row["Year"])
        return (row[col] / pop) * 1e5 if pop else np.nan

    df["Cardiovascular_per_100k"] = df.apply(lambda r: per_100k(r, "Cardiovascular"), axis=1)
    df["Respiratory_per_100k"] = df.apply(lambda r: per_100k(r, "Lower_Respiratory" + ""), axis=1)
    df["ChronicResp_per_100k"] = df.apply(lambda r: per_100k(r, "Chronic_Respiratory"), axis=1)
    df["All_Respiratory_per_100k"] = df["Respiratory_per_100k"] + df["ChronicResp_per_100k"]
    df["All_Key_Diseases_per_100k"] = df["Cardiovascular_per_100k"] + df["All_Respiratory_per_100k"]

    national_rates = df.groupby("Year").agg({
        "Cardiovascular_per_100k": "mean",
        "All_Respiratory_per_100k": "mean",
        "All_Key_Diseases_per_100k": "mean",
    }).rename(columns=lambda c: f"national_{c}")

    return df[["Year", "Cardiovascular_per_100k", "All_Respiratory_per_100k", "All_Key_Diseases_per_100k"]], national_rates


def estimate_state_rates(state_df: pd.DataFrame, national_rates: pd.DataFrame) -> pd.DataFrame:
    merged = state_df.merge(national_rates, left_on="Year", right_index=True, how="left")
    national_mean_aqi = merged["mean_AQI"].mean()
    scale = (merged["mean_AQI"] / national_mean_aqi) ** 1.5
    merged["Cardiovascular_per_100k"] = merged["national_Cardiovascular_per_100k"] * scale
    merged["Respiratory_per_100k"] = merged["national_All_Respiratory_per_100k"] * scale
    merged["All_Key_Diseases_per_100k"] = merged["national_All_Key_Diseases_per_100k"] * scale
    return merged.drop(columns=["national_Cardiovascular_per_100k", "national_All_Respiratory_per_100k", "national_All_Key_Diseases_per_100k"])


def prepare_model3_data(city_path: str = "city_day.csv", global_path: str = "global_air_pollution_data.csv", deaths_path: str = "cause_of_deaths.csv", output_path: str = "model3_disease_burden.csv") -> Path:
    # Approximate India population (World Bank, billions) for 2015-2019
    population_map = {
        2015: 1_311_000_000,
        2016: 1_324_000_000,
        2017: 1_339_000_000,
        2018: 1_354_000_000,
        2019: 1_368_000_000,
    }

    city_df = load_city_day(city_path)
    state_year = compute_state_year_agg(city_df)

    global_poll = load_global_pollution(global_path)
    disease_df, national_rates = load_disease_data(deaths_path, population_map)

    # Merge pollution (state-year) with global India pollution (state) and disease estimates
    merged = state_year.merge(global_poll, on="State", how="left", suffixes=('', '_global'))
    merged = merged.merge(disease_df, on="Year", how="left")

    merged = estimate_state_rates(merged, national_rates)

    # Interaction features
    merged["PM2.5_SO2"] = merged["PM2.5"] * merged["SO2"]
    merged["PM2.5_NO2"] = merged["PM2.5"] * merged["NO2"]
    merged["AQI_pct_severe"] = merged["mean_AQI"] * merged["pct_severe_days"]

    # Select and order columns
    columns = [
        "State",
        "Year",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "NOx",
        "mean_AQI",
        "max_AQI",
        "std_AQI",
        "pct_severe_days",
        "pct_very_poor_days",
        "PM2.5_value",
        "NO2_value",
        "Ozone_value",
        "AQI_value",
        "CO_value",
        "PM2.5_SO2",
        "PM2.5_NO2",
        "AQI_pct_severe",
        "Cardiovascular_per_100k",
        "Respiratory_per_100k",
        "All_Key_Diseases_per_100k",
    ]

    for col in columns:
        if col not in merged.columns:
            merged[col] = np.nan

    df_final = merged[columns]

    # Median imputation for numeric columns
    num_cols = df_final.select_dtypes(include=[np.number]).columns
    medians = df_final[num_cols].median()
    df_final[num_cols] = df_final[num_cols].fillna(medians)

    df_final.to_csv(output_path, index=False)

    print(f"Saved {output_path} with {len(df_final)} rows and {len(df_final.columns)} columns.")
    print("Columns:", df_final.columns.tolist())
    return Path(output_path)


if __name__ == "__main__":
    prepare_model3_data()
