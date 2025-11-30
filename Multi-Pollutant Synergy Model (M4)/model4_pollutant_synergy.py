import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TARGETS = [
    "Cardiovascular_deaths_per_100k",
    "Respiratory_deaths_per_100k",
    "Combined_disease_risk_score",
]


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, r2, mape


def build_preprocessor(feature_df: pd.DataFrame):
    categorical = ["Country"] if "Country" in feature_df.columns else []
    numeric = [c for c in feature_df.columns if c not in categorical + ["State"]]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipe, categorical),
            ("numeric", num_pipe, numeric),
        ],
        remainder="drop",
    )
    return preprocessor


def get_models():
    models = [
        ("Ridge", Ridge(), {"model__alpha": [1.0, 5.0, 10.0]}),
        ("Lasso", Lasso(max_iter=5000), {"model__alpha": [0.1, 0.5, 1.0]}),
        (
            "ElasticNet",
            ElasticNet(max_iter=5000),
            {"model__alpha": [0.1, 0.5, 1.0], "model__l1_ratio": [0.3, 0.5, 0.7]},
        ),
        (
            "GradientBoosting",
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [150, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        ),
        (
            "RandomForest",
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [8, None], "model__min_samples_leaf": [2, 5]},
        ),
    ]
    return models


def plot_predictions(y_true, y_pred, out_file: Path, title: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def train_target(df: pd.DataFrame, target: str, out_dir: Path):
    df = df.dropna(subset=[target]).copy()

    y_raw = df[target]
    if y_raw.nunique() <= 1:
        print(f"Target {target} has no variance; skipping.")
        return None

    y = np.log1p(y_raw)
    # Restrict features to core pollutants and key interactions
    base_feats = ["PM2.5", "NO2", "SO2", "CO", "Ozone"]
    interaction_feats = ["PM25_NO2", "PM25_SO2", "PM25_CO", "NO2_SO2", "SO2_CO"]
    keep_cols = [c for c in base_feats + interaction_feats if c in df.columns]
    X = df[keep_cols + ["Country"]] if "Country" in df.columns else df[keep_cols]

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    groups = df["Country"].fillna("unknown") if "Country" in df.columns else None
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    preprocessor = build_preprocessor(X)
    models = get_models()

    results = []
    best = None

    for name, estimator, param_grid in models:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        use_random = name in {"RandomForest", "GradientBoosting"}
        search_cls = RandomizedSearchCV if use_random else GridSearchCV
        search_params = {
            "estimator": pipe,
            "cv": 3,
            "scoring": "r2",
            "n_jobs": -1,
            "verbose": 0,
        }
        if param_grid:
            if use_random:
                search_params.update({"param_distributions": param_grid, "n_iter": min(6, sum(len(v) for v in param_grid.values())), "random_state": 42})
            else:
                search_params.update({"param_grid": param_grid})
        else:
            search_params.update({"param_grid": {"model": [estimator]}})

        start = time.time()
        search = search_cls(**search_params)
        search.fit(X_train, y_train)
        duration = time.time() - start

        best_est = search.best_estimator_
        y_pred_train_log = best_est.predict(X_train)
        y_pred_test_log = best_est.predict(X_test)

        train_rmse, train_mae, train_r2, train_mape = regression_metrics(y_train, y_pred_train_log)
        test_rmse, test_mae, test_r2, test_mape = regression_metrics(y_test, y_pred_test_log)
        gap = train_r2 - test_r2

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
                "Gap": gap,
                "CV_Best_Score": search.best_score_,
                "Best_Params": search.best_params_,
                "Training_Time": duration,
                "Best_Estimator": best_est,
                "Test_Preds": np.expm1(y_pred_test_log),
                "Test_True": np.expm1(y_test),
            }
        )

        if best is None or test_r2 > best["Test_R2"]:
            best = results[-1]

        print(f"Target {target} | Completed {name}: Test R2 {test_r2:.3f}, RMSE(log) {test_rmse:.3f}")

    if best is None:
        print(f"No model trained for {target}")
        return None

    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(["Test_R2", "Test_RMSE"], ascending=[False, True])
    results_df_sorted.drop(columns=["Best_Estimator", "Test_Preds", "Test_True"], inplace=True)
    results_df_sorted.to_csv(out_dir / f"model4_{target}_comparison.csv", index=False)

    best_estimator = best["Best_Estimator"]
    best_name = best["Model_Name"]
    best_r2 = best["Test_R2"]

    for old in out_dir.glob(f"model4_best_{target}_*.pkl"):
        try:
            old.unlink()
        except Exception:
            pass
    model_filename = f"model4_best_{target}_{best_name}_R2-{best_r2:.3f}.pkl"
    joblib.dump(best_estimator, out_dir / model_filename)

    preds = best["Test_Preds"]
    true_vals = best["Test_True"]
    preds_df = pd.DataFrame(
        {
            "Country": pd.Series(X_test.get("Country", pd.Series(["unknown"] * len(X_test)))).reset_index(drop=True),
            "State": pd.Series(X_test.get("State", pd.Series(["unknown"] * len(X_test)))).reset_index(drop=True),
            "Year": pd.Series(X_test.get("Year", pd.Series([np.nan] * len(X_test)))).reset_index(drop=True),
            f"Actual_{target}": pd.Series(true_vals).reset_index(drop=True),
            f"Pred_{target}": pd.Series(preds).reset_index(drop=True),
        }
    )
    preds_df.to_csv(out_dir / f"model4_{target}_predictions.csv", index=False)

    plot_predictions(true_vals, preds, out_dir / f"model4_{target}_actual_vs_pred.png", f"{target} - Actual vs Pred")

    return {
        "target": target,
        "best_name": best_name,
        "best_r2": best_r2,
        "best_rmse": best["Test_RMSE"],
        "model_file": model_filename,
        "comparison_file": f"model4_{target}_comparison.csv",
        "pred_file": f"model4_{target}_predictions.csv",
    }


def main(input_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    summaries = []
    for tgt in TARGETS:
        result = train_target(df, tgt, out_dir)
        if result:
            summaries.append(result)

    pd.DataFrame(summaries).to_csv(out_dir / "model4_summary.csv", index=False)
    print("\nModel 4 training complete. Summary:")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-pollutant synergy models")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "model4_pollutant_synergy.csv"),
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
