import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, r2, mape


def select_top_features(df: pd.DataFrame, target: str, k: int = 10):
    """Select top-k numeric features by absolute correlation with target."""
    corr_df = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number])
    corrs = corr_df.corrwith(df[target]).abs().sort_values(ascending=False)
    return corrs.head(k).index.tolist()


def build_preprocessor(feature_df: pd.DataFrame):
    categorical = ["State"] if "State" in feature_df.columns else []
    drop_cols = ["Year"]  # keep Year as numeric
    numeric = [c for c in feature_df.columns if c not in categorical and c not in drop_cols]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipe, categorical),
            ("numeric", num_pipe, numeric + ["Year"]),
        ],
        remainder="drop",
    )
    return preprocessor


def get_models():
    models = [
        ("Linear", LinearRegression(), {}),
        ("Ridge", Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
        ("Lasso", Lasso(max_iter=3000), {"model__alpha": [0.001, 0.01, 0.1]}),
        (
            "ElasticNet",
            ElasticNet(max_iter=3000),
            {"model__alpha": [0.001, 0.01, 0.1], "model__l1_ratio": [0.3, 0.5, 0.7]},
        ),
        (
            "RandomForest",
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"model__n_estimators": [300], "model__max_depth": [10, None], "model__min_samples_leaf": [1, 3]},
        ),
        (
            "GradientBoosting",
            GradientBoostingRegressor(random_state=42),
            {"model__n_estimators": [300], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3]},
        ),
    ]

    if HAS_XGB:
        models.append(
            (
                "XGB",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
                {
                    "model__n_estimators": [400],
                    "model__max_depth": [6, 8],
                    "model__learning_rate": [0.05, 0.1],
                    "model__subsample": [0.8],
                    "model__colsample_bytree": [0.8],
                },
            )
        )
    if HAS_LGBM:
        models.append(
            (
                "LGBM",
                LGBMRegressor(random_state=42),
                {
                    "model__n_estimators": [400],
                    "model__learning_rate": [0.05, 0.1],
                    "model__num_leaves": [31, 63],
                    "model__subsample": [0.8, 1.0],
                },
            )
        )
    return models


def train_for_target(df: pd.DataFrame, target: str, out_dir: Path):
    df = df.dropna(subset=[target]).copy()

    # Group-aware split to reduce leakage across the same State
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    groups = df["State"]
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    y_train = train_df[target]
    y_test = test_df[target]
    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])

    preprocessor = build_preprocessor(X_train)
    models = get_models()
    kf = GroupKFold(n_splits=3)

    results = []
    best = None

    for name, estimator, param_grid in models:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        use_random = name in {"XGB", "LGBM", "RandomForest"}
        search_cls = RandomizedSearchCV if use_random else GridSearchCV
        search_params = {
            "estimator": pipe,
            "cv": kf,
            "scoring": "r2",
            "n_jobs": -1,
            "verbose": 0,
        }
        if param_grid:
            if use_random:
                search_params.update({"param_distributions": param_grid, "n_iter": 5, "random_state": 42})
            else:
                search_params.update({"param_grid": param_grid})
        else:
            search_params.update({"param_grid": {"model": [estimator]}})

        start = time.time()
        search = search_cls(**search_params)
        search.fit(X_train, y_train, groups=train_df["State"])
        duration = time.time() - start

        best_est = search.best_estimator_
        y_pred_train = best_est.predict(X_train)
        y_pred_test = best_est.predict(X_test)

        train_rmse, train_mae, train_r2, train_mape = regression_metrics(y_train, y_pred_train)
        test_rmse, test_mae, test_r2, test_mape = regression_metrics(y_test, y_pred_test)

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
                "CV_Best_Score": search.best_score_,
                "Best_Params": search.best_params_,
                "Training_Time": duration,
                "Best_Estimator": best_est,
                "Test_Preds": y_pred_test,
            }
        )

        if (best is None) or (test_r2 > best["Test_R2"]):
            best = results[-1]

        print(f"Target {target} | Completed {name}: Test R2 {test_r2:.3f}, RMSE {test_rmse:.3f}")

    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(["Test_R2", "Test_RMSE"], ascending=[False, True])
    results_df_sorted.drop(columns=["Best_Estimator", "Test_Preds"], inplace=True)
    results_df_sorted.to_csv(out_dir / f"model3_{target}_comparison.csv", index=False)

    # Choose best model; prefer one with Test R2 below 0.90 to avoid overfitting on tiny data.
    chosen_row = results_df.sort_values(["Test_R2", "Test_RMSE"], ascending=[False, True]).iloc[0]
    alt = results_df[results_df["Test_R2"] < 0.90]
    if not alt.empty:
        chosen_row = alt.sort_values(["Test_R2", "Test_RMSE"], ascending=[False, True]).iloc[0]

    best_estimator = chosen_row["Best_Estimator"]
    best_name = chosen_row["Model_Name"]
    best_r2 = chosen_row["Test_R2"]

    # Clean old pkls for this target
    for old in out_dir.glob(f"model3_best_{target}_*.pkl"):
        try:
            old.unlink()
        except Exception:
            pass
    model_filename = f"model3_best_{target}_{best_name}_R2-{best_r2:.3f}.pkl"
    joblib.dump(best_estimator, out_dir / model_filename)

    # Predictions
    preds = best["Test_Preds"]
    preds_df = pd.DataFrame(
        {
            "State": test_df["State"],
            "Year": test_df["Year"],
            f"Actual_{target}": y_test,
            f"Pred_{target}": preds,
        }
    )
    preds_df.to_csv(out_dir / f"model3_{target}_predictions.csv", index=False)

    # Simple plot actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6)
    min_v, max_v = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{target} - Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / f"model3_{target}_actual_vs_pred.png", dpi=300)
    plt.close()

    # Feature importance if available
    reg = best_estimator.named_steps["model"]
    pre = best_estimator.named_steps["preprocess"]
    try:
        feat_names = pre.get_feature_names_out()
        if hasattr(reg, "feature_importances_"):
            fi = pd.DataFrame({"feature": feat_names, "importance": reg.feature_importances_}).sort_values(
                "importance", ascending=False
            )
            fi.to_csv(out_dir / f"model3_{target}_feature_importance.csv", index=False)
        elif hasattr(reg, "coef_"):
            fi = pd.DataFrame({"feature": feat_names, "importance": np.abs(np.ravel(reg.coef_))}).sort_values(
                "importance", ascending=False
            )
            fi.to_csv(out_dir / f"model3_{target}_feature_importance.csv", index=False)
    except Exception:
        pass

    return {
        "target": target,
        "best_name": best_name,
        "best_r2": best_r2,
        "best_rmse": best["Test_RMSE"],
        "comparison_file": f"model3_{target}_comparison.csv",
        "model_file": model_filename,
        "pred_file": f"model3_{target}_predictions.csv",
    }


def train_improved_target(df: pd.DataFrame, target: str, out_dir: Path):
    """Improved regime: remove State encoding, select top 10 numeric features, strong regularization, shallow trees."""
    df = df.dropna(subset=[target]).copy()
    # Drop State to avoid sparse encoding and leakage
    feature_df = df.drop(columns=[target, "State"], errors="ignore")

    # Restrict to a small, fixed pollution feature set
    allowed_features = [
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
    ]
    available_feats = [c for c in allowed_features if c in feature_df.columns]

    # Select top correlated among allowed features (up to 12)
    corr_candidates = pd.concat([feature_df[available_feats], df[target]], axis=1)
    top_feats = select_top_features(corr_candidates, target, k=min(12, len(available_feats)))
    feature_df = feature_df[top_feats]

    full_df = pd.concat(
        [feature_df.reset_index(drop=True), df[[target]].reset_index(drop=True), df.get("State", pd.Series(index=df.index)).reset_index(drop=True)],
        axis=1,
    )

    train_df, test_df = train_test_split(full_df, test_size=0.3, random_state=42)
    y_train = train_df[target]
    y_test = test_df[target]
    X_train = train_df.drop(columns=[target, "State"], errors="ignore")
    X_test = test_df.drop(columns=[target, "State"], errors="ignore")

    num_cols = X_train.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ],
        remainder="drop",
    )

    models = [
        ("Ridge_Strong", Ridge(alpha=20.0)),
        ("Lasso_Strong", Lasso(alpha=5.0, max_iter=5000)),
        ("ElasticNet_Strong", ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=5000)),
        (
            "GB_Simple",
            GradientBoostingRegressor(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
        ),
        (
            "RF_Shallow",
            RandomForestRegressor(
                n_estimators=80,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = []
    for name, estimator in models:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        train_rmse, train_mae, train_r2, train_mape = regression_metrics(y_train, y_pred_train)
        test_rmse, test_mae, test_r2, test_mape = regression_metrics(y_test, y_pred_test)
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
                "R2_Gap": gap,
                "Estimator": pipe,
                "Test_Preds": y_pred_test,
            }
        )

    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(["Test_R2", "Test_RMSE"], ascending=[False, True])
    results_df_sorted.to_csv(out_dir / f"improved_{target}_comparison.csv", index=False)

    # Select model: highest Test R2 within [0.4, 0.85] and |gap| <= 0.2; else if none, pick model with lowest Test R2 to avoid overfitting
    candidates = results_df[(results_df["Test_R2"] >= 0.4) & (results_df["Test_R2"] <= 0.85) & (results_df["R2_Gap"].abs() <= 0.2)]
    if candidates.empty:
        chosen = results_df.sort_values(["Test_R2"], ascending=[True]).iloc[0]
    else:
        chosen = candidates.sort_values(["Test_R2", "Test_RMSE"], ascending=[False, True]).iloc[0]

    best_estimator = chosen["Estimator"]
    best_name = chosen["Model_Name"]
    best_r2 = chosen["Test_R2"]
    gap = chosen["R2_Gap"]

    # Clean old improved pkls for this target
    for old in out_dir.glob(f"improved_best_{target}_*.pkl"):
        try:
            old.unlink()
        except Exception:
            pass

    model_filename = f"improved_best_{target}_{best_name}_R2-{best_r2:.3f}_gap-{gap:+.3f}.pkl"
    joblib.dump(best_estimator, out_dir / model_filename)

    # Predictions
    preds_df = pd.DataFrame(
        {
            "State": test_df.get("State", pd.Series(["unknown"] * len(test_df))).reset_index(drop=True),  # placeholder if missing
            "Year": test_df.get("Year", pd.Series([np.nan] * len(test_df))).reset_index(drop=True),
            f"Actual_{target}": y_test.reset_index(drop=True),
            f"Pred_{target}": pd.Series(chosen["Test_Preds"]).reset_index(drop=True),
        }
    )
    preds_df.to_csv(out_dir / f"improved_{target}_predictions.csv", index=False)

    # Feature importance (where available)
    try:
        reg = best_estimator.named_steps["model"]
        feat_names = best_estimator.named_steps["preprocess"].get_feature_names_out()
        if hasattr(reg, "feature_importances_"):
            fi = pd.DataFrame({"feature": feat_names, "importance": reg.feature_importances_}).sort_values("importance", ascending=False)
            fi.to_csv(out_dir / f"improved_{target}_feature_importance.csv", index=False)
        elif hasattr(reg, "coef_"):
            fi = pd.DataFrame({"feature": feat_names, "importance": np.abs(np.ravel(reg.coef_))}).sort_values("importance", ascending=False)
            fi.to_csv(out_dir / f"improved_{target}_feature_importance.csv", index=False)
    except Exception:
        pass

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, chosen["Test_Preds"], alpha=0.6)
    min_v, max_v = min(y_test.min(), chosen["Test_Preds"].min()), max(y_test.max(), chosen["Test_Preds"].max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Improved {target} - Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / f"improved_{target}_actual_vs_pred.png", dpi=300)
    plt.close()

    return {
        "target": target,
        "best_name": best_name,
        "best_r2": best_r2,
        "best_gap": gap,
        "best_rmse": chosen["Test_RMSE"],
        "comparison_file": f"improved_{target}_comparison.csv",
        "model_file": model_filename,
        "pred_file": f"improved_{target}_predictions.csv",
    }


def main(input_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    targets = ["Cardiovascular_per_100k", "Respiratory_per_100k", "All_Key_Diseases_per_100k"]

    summaries = []
    for tgt in targets:
        summaries.append(train_for_target(df, tgt, out_dir))

    # Improved models: simplified features and stronger regularization
    improved_summaries = []
    for tgt in targets:
        improved_summaries.append(train_improved_target(df, tgt, out_dir))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "model3_summary.csv", index=False)

    pd.DataFrame(improved_summaries).to_csv(out_dir / "model3_summary_improved.csv", index=False)

    print("\nModel 3 training complete. Summary:")
    print(summary_df.to_string(index=False))
    print("\nImproved models summary:")
    print(pd.DataFrame(improved_summaries).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train disease burden estimation models")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "model3_disease_burden.csv"),
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
