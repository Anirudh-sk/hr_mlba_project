import pandas as pd
import numpy as np
from pathlib import Path


def month_to_season(month: int) -> int:
    """Map month to season code: 1=winter, 2=spring, 3=summer, 4=monsoon."""
    if month in (11, 12, 1, 2):
        return 1
    if month in (3, 4):
        return 2
    if month in (5, 6):
        return 3
    return 4  # Jul-Oct


def compute_days_since_last_severe(severe_series: pd.Series) -> pd.Series:
    """Compute days since last severe day within each city.
    Returns NaN until a severe day has occurred. 0 on severe days.
    """
    arr = severe_series.to_numpy(dtype=float)
    positions = np.arange(len(arr), dtype=float)
    last_pos = np.where(arr == 1, positions, np.nan)
    last_pos_ffill = pd.Series(last_pos).ffill().to_numpy()
    days_since = positions - last_pos_ffill
    days_since[np.isnan(last_pos_ffill)] = np.nan
    return pd.Series(days_since, index=severe_series.index)


def prepare_model2_data(input_path: str = "city_day.csv", output_path: str = "model2_severe_day.csv") -> Path:
    """Prepare features for severe pollution day prediction (AQI >= 300)."""
    required_cols = {
        "City",
        "Date",
        "AQI",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "NO",
        "NOx",
    }

    csv_path = Path(input_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    group = df.groupby("City", group_keys=False)

    lag_features = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NO", "NOx"]
    for col in lag_features:
        for lag in (1, 2, 3):
            df[f"{col}_lag_{lag}"] = group[col].shift(lag)

    # 3-day rolling stats for AQI and major pollutants
    for col in lag_features:
        rolling = group[col].rolling(window=3, min_periods=3)
        df[f"{col}_rolling_mean_3"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"{col}_rolling_max_3"] = rolling.max().reset_index(level=0, drop=True)
        df[f"{col}_rolling_std_3"] = rolling.std().reset_index(level=0, drop=True)

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["season"] = df["month"].apply(month_to_season)
    df["is_winter"] = df["month"].isin([11, 12, 1]).astype(int)

    # Rate of change features
    df["AQI_change_1d"] = df["AQI"] - df["AQI_lag_1"]
    df["AQI_change_3d"] = df["AQI"] - df["AQI_lag_3"]
    df["PM2.5_change_1d"] = df["PM2.5"] - df["PM2.5_lag_1"]
    df["PM10_change_1d"] = df["PM10"] - df["PM10_lag_1"]

    df["severe_today"] = (df["AQI"] >= 300).astype(int)
    df["was_severe_yesterday"] = (df["AQI_lag_1"] >= 300).astype(float)

    # Days since last severe day (per city)
    df["days_since_last_severe"] = group["severe_today"].apply(compute_days_since_last_severe)

    # Target: is severe tomorrow (shift -1)
    df["is_severe_tomorrow"] = group["severe_today"].shift(-1)

    feature_cols = [
        "City",
        "Date",
        # lags
        *[f"{col}_lag_{lag}" for col in lag_features for lag in (1, 2, 3)],
        # rolling stats
        *[f"{col}_rolling_mean_3" for col in lag_features],
        *[f"{col}_rolling_max_3" for col in lag_features],
        *[f"{col}_rolling_std_3" for col in lag_features],
        # temporal
        "day_of_week",
        "month",
        "season",
        "is_winter",
        # deltas
        "AQI_change_1d",
        "AQI_change_3d",
        "PM2.5_change_1d",
        "PM10_change_1d",
        # categorical features
        "was_severe_yesterday",
        "days_since_last_severe",
        # target
        "is_severe_tomorrow",
    ]

    drop_subset = [col for col in feature_cols if col not in {"City", "Date"}]
    before_drop = len(df)
    df_final = df.dropna(subset=drop_subset).copy()
    after_drop = len(df_final)

    df_final = df_final[feature_cols]
    df_final.to_csv(output_path, index=False)

    # Class distribution
    severe_counts = df_final["is_severe_tomorrow"].value_counts().to_dict()
    severe_pct = {k: round(v / len(df_final) * 100, 2) for k, v in severe_counts.items()}

    print(f"Saved {output_path} with {after_drop} rows (dropped {before_drop - after_drop}).")
    print(f"Severe day distribution (is_severe_tomorrow): {severe_counts} | pct: {severe_pct}")
    print(f"Columns: {len(df_final.columns)} -> {df_final.columns.tolist()}")
    return Path(output_path)


if __name__ == "__main__":
    prepare_model2_data()
