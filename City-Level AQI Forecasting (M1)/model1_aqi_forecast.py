import argparse
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone

warnings.filterwarnings("ignore")

# Optional imports
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


METRICS = ["RMSE", "MAE", "R2", "MAPE"]


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, r2, mape


def build_preprocessor(feature_df):
    categorical_features = ["City"] if "City" in feature_df.columns else []
    numeric_features = [c for c in feature_df.columns if c not in categorical_features + ["Date", "AQI_target"]]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def get_models():
    models = [
        ("Lasso", Lasso(max_iter=5000), {"model__alpha": [0.05, 0.1, 0.5]}),
        (
            "RandomForest",
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [250],
                "model__max_depth": [None],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [300],
                "model__learning_rate": [0.1],
                "model__max_depth": [3],
                "model__subsample": [0.9],
            },
        ),
    ]

    if HAS_XGB:
        models.append(
            (
                "XGBRegressor",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                    verbosity=0,
                ),
                {
                    "model__n_estimators": [350],
                    "model__max_depth": [8],
                    "model__learning_rate": [0.08],
                    "model__subsample": [0.8],
                    "model__colsample_bytree": [0.8],
                    "model__min_child_weight": [1],
                },
            )
        )
    if HAS_LGBM:
        models.append(
            (
                "LGBMRegressor",
                LGBMRegressor(random_state=42),
                {
                    "model__n_estimators": [350],
                    "model__learning_rate": [0.08],
                    "model__num_leaves": [63],
                    "model__subsample": [0.8],
                },
            )
        )

    # Quantile regression variant to better capture extremes
    models.append(
        (
            "GBR_Quantile",
            GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=42),
            {"model__n_estimators": [300], "model__learning_rate": [0.08], "model__max_depth": [3]},
        )
    )
    return models


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-linear and extreme-event-focused features."""
    df = df.copy()

    # Non-linear transforms
    if "AQI_lag_1" in df.columns:
        df["AQI_lag_1_squared"] = df["AQI_lag_1"] ** 2
        df["AQI_lag_1_log"] = np.log1p(df["AQI_lag_1"].clip(lower=0))

    # Extreme indicators
    df["was_severe_last_week"] = (df.get("AQI_lag_7", 0) > 300).astype(int)

    # Count of high AQI days in last week using available lags
    lag_cols = [c for c in df.columns if c.startswith("AQI_lag_")]
    high_counts = np.zeros(len(df))
    for c in lag_cols:
        high_counts += (df[c] > 300).astype(int)
    df["high_days_last_week"] = high_counts

    # Interaction with season
    if "PM2.5_lag_1" in df.columns and "is_winter" in df.columns:
        df["PM25_winter_interaction"] = df["PM2.5_lag_1"] * df["is_winter"]

    return df


def plot_predictions(y_true, y_pred, dates, out_dir: Path, model_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "model1_actual_vs_predicted.png", dpi=300)
    plt.close()

    # Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted AQI")
    plt.ylabel("Residual")
    plt.title(f"{model_name} - Residuals")
    plt.tight_layout()
    plt.savefig(out_dir / "model1_residuals.png", dpi=300)
    plt.close()

    # Time series (test set)
    if dates is not None:
        order = np.argsort(dates)
        plt.figure(figsize=(10, 4))
        plt.plot(np.array(dates)[order], np.array(y_true)[order], label="Actual")
        plt.plot(np.array(dates)[order], np.array(y_pred)[order], label="Predicted")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.title(f"{model_name} - Time Series (Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "model1_time_series.png", dpi=300)
        plt.close()


def get_feature_importance(model, feature_names):
    reg = model
    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    if hasattr(reg, "coef_"):
        coefs = np.ravel(reg.coef_)
        return pd.DataFrame({"feature": feature_names, "importance": np.abs(coefs)}).sort_values("importance", ascending=False)
    return pd.DataFrame(columns=["feature", "importance"])


def main(input_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "AQI_target"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = add_extra_features(df)

    y = df["AQI_target"]
    feature_df = df.drop(columns=["AQI_target"])

    preprocessor, numeric_features, categorical_features = build_preprocessor(feature_df)
    models = get_models()
    tscv = TimeSeriesSplit(n_splits=2)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = feature_df.iloc[:split_idx], feature_df.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    results = []
    best_overall = None

    for name, estimator, param_grid in models:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

        use_random = name in {"XGBRegressor", "LGBMRegressor", "RandomForest"}
        search_cls = RandomizedSearchCV if use_random else GridSearchCV
        search_params = {
            "estimator": pipe,
            "cv": tscv,
            "scoring": "neg_mean_squared_error",
            "n_jobs": -1,
            "verbose": 0,
        }
        if param_grid:
            if use_random:
                search_params.update({"param_distributions": param_grid, "n_iter": 3, "random_state": 42})
            else:
                search_params.update({"param_grid": param_grid})
        else:
            search_params.update({"param_grid": {"model": [estimator]}})

        search = search_cls(**search_params)

        start = time.time()
        search.fit(X_train, y_train)
        duration = time.time() - start

        best_model = search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        train_rmse, train_mae, train_r2, train_mape = regression_metrics(y_train, y_pred_train)
        test_rmse, test_mae, test_r2, test_mape = regression_metrics(y_test, y_pred_test)

        cv_rmse = np.sqrt(-search.best_score_)
        # best_score_std is not directly available; approximate from cv_results_
        mask = search.cv_results_["rank_test_score"] == 1
        std_scores = search.cv_results_["std_test_score"][mask]
        cv_rmse_std = np.sqrt(-std_scores[0]) if len(std_scores) else np.nan

        results.append(
            {
                "Model_Name": name,
                "Train_RMSE": train_rmse,
                "Test_RMSE": test_rmse,
                "Train_MAE": train_mae,
                "Test_MAE": test_mae,
                "Train_R2": train_r2,
                "Test_R2": test_r2,
                "Train_MAPE": train_mape,
                "Test_MAPE": test_mape,
                "CV_RMSE_Mean": cv_rmse,
                "CV_RMSE_Std": cv_rmse_std,
                "Training_Time": duration,
                "Best_Params": search.best_params_,
                "Best_Estimator": best_model,
            }
        )

        if (best_overall is None) or (test_rmse < best_overall["Test_RMSE"]):
            best_overall = results[-1]

        print(f"Completed {name}: Test RMSE {test_rmse:.3f}, Test R2 {test_r2:.3f}, CV RMSE {cv_rmse:.3f}")

    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(["Test_RMSE", "Test_R2"], ascending=[True, False])
    results_df_sorted.drop(columns=["Best_Estimator"], inplace=True)
    results_df_sorted.to_csv(out_dir / "model1_comparison.csv", index=False)

    best_model_eval = best_overall["Best_Estimator"]
    best_name = best_overall["Model_Name"]
    best_r2 = best_overall["Test_R2"]

    y_pred_test = best_model_eval.predict(X_test)
    test_dates = df.iloc[split_idx:]["Date"]
    test_cities = df.iloc[split_idx:]["City"]

    predictions_df = pd.DataFrame(
        {
            "Date": test_dates,
            "City": test_cities,
            "Actual_AQI": y_test,
            "Predicted_AQI": y_pred_test,
        }
    )
    predictions_df.to_csv(out_dir / "model1_predictions.csv", index=False)

    # Refit best model on full dataset for export
    final_model = clone(best_model_eval)
    final_model.fit(feature_df, y)

    # Clean old model artifacts and save single best pipeline with metric in name
    for old_pkl in out_dir.glob("model1_*best*.pkl"):
        try:
            old_pkl.unlink()
        except Exception:
            pass
    for extra in ["model1_preprocessor.pkl", "model1_features.pkl"]:
        extra_path = out_dir / extra
        if extra_path.exists():
            try:
                extra_path.unlink()
            except Exception:
                pass
    model_filename = f"model1_best_{best_name}_R2-{best_r2:.3f}.pkl"
    joblib.dump(final_model, out_dir / model_filename)

    feature_names = final_model.named_steps["preprocess"].get_feature_names_out()

    # Plots
    plot_predictions(y_test.to_numpy(), y_pred_test, test_dates.to_numpy(), out_dir, best_overall["Model_Name"])

    # Feature importance (if available)
    try:
        reg = final_model.named_steps["model"]
        fi = get_feature_importance(reg, feature_names)
        if not fi.empty:
            fi.head(30).to_csv(out_dir / "model1_feature_importance.csv", index=False)
    except Exception:
        pass

    # Final summary
    print("\nBest model:", best_overall["Model_Name"])
    print("Params:", best_overall["Best_Params"])
    print(
        f"Test RMSE: {best_overall['Test_RMSE']:.3f}, Test MAE: {best_overall['Test_MAE']:.3f}, Test R2: {best_overall['Test_R2']:.3f}, Test MAPE: {best_overall['Test_MAPE']:.3f}"
    )
    print(f"Comparison table saved to {out_dir / 'model1_comparison.csv'}")
    print(f"Predictions saved to {out_dir / 'model1_predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AQI forecasting models")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "model1_aqi_forecast.csv"),
        help="Path to prepared dataset",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory to save outputs",
    )
    args = parser.parse_args()
    main(args.input, args.outdir)
