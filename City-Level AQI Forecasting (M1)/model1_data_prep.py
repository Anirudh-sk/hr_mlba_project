import pandas as pd
from pathlib import Path


def month_to_season(month: int) -> int:
    """Map month to season code: 1=winter, 2=spring, 3=summer, 4=monsoon.
    Winter uses Nov-Feb to align with the is_winter flag.
    """
    if month in (11, 12, 1, 2):
        return 1
    if month in (3, 4):
        return 2
    if month in (5, 6):
        return 3
    return 4  # Jul-Oct treated as monsoon/post-monsoon


def prepare_model1_data(input_path: str = "city_day.csv", output_path: str = "model1_aqi_forecast.csv") -> Path:
    """Create lagged/rolling features and a 7-day-ahead target for AQI forecasting."""
    required_cols = {"City", "Date", "AQI", "PM2.5", "PM10", "NO2", "SO2"}
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

    lag_features = ["AQI", "PM2.5", "PM10", "NO2", "SO2"]
    for col in lag_features:
        for lag in range(1, 8):
            df[f"{col}_lag_{lag}"] = group[col].shift(lag)

    df["AQI_rolling_mean_7"] = group["AQI"].rolling(window=7, min_periods=7).mean().reset_index(level=0, drop=True)
    df["AQI_rolling_std_7"] = group["AQI"].rolling(window=7, min_periods=7).std().reset_index(level=0, drop=True)
    df["AQI_rolling_max_7"] = group["AQI"].rolling(window=7, min_periods=7).max().reset_index(level=0, drop=True)
    df["AQI_rolling_min_7"] = group["AQI"].rolling(window=7, min_periods=7).min().reset_index(level=0, drop=True)

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["season"] = df["month"].apply(month_to_season)
    df["is_winter"] = df["month"].isin([11, 12, 1]).astype(int)

    df["AQI_ema_7"] = group["AQI"].apply(lambda s: s.ewm(alpha=0.3, adjust=False).mean())

    df["AQI_target"] = group["AQI"].shift(-7)

    feature_cols = [
        "City",
        "Date",
        *[f"{col}_lag_{lag}" for col in lag_features for lag in range(1, 8)],
        "AQI_rolling_mean_7",
        "AQI_rolling_std_7",
        "AQI_rolling_max_7",
        "AQI_rolling_min_7",
        "day_of_week",
        "month",
        "season",
        "is_winter",
        "AQI_ema_7",
        "AQI_target",
    ]

    base_keep = ["AQI", "PM2.5", "PM10", "NO2", "SO2"]
    final_cols = ["City", "Date", *base_keep] + [col for col in feature_cols if col not in {"City", "Date"}]

    drop_subset = [col for col in feature_cols if col not in {"City", "Date"}]
    before_drop = len(df)
    df_final = df.dropna(subset=drop_subset).copy()
    after_drop = len(df_final)

    df_final = df_final[final_cols]
    df_final.to_csv(output_path, index=False)

    print(f"Saved {output_path} with {after_drop} rows (dropped {before_drop - after_drop}).")
    print(f"Columns: {len(df_final.columns)} -> {df_final.columns.tolist()}")
    return Path(output_path)


if __name__ == "__main__":
    prepare_model1_data()
