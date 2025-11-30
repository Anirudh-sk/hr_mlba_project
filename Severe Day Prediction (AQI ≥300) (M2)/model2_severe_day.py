import argparse
import time
import warnings
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout for paths with special characters
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier  # type: ignore

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def build_preprocessor(feature_df: pd.DataFrame):
    categorical_features = ["City"] if "City" in feature_df.columns else []
    drop_cols = ["Date", "is_severe_tomorrow"]
    numeric_features = [c for c in feature_df.columns if c not in categorical_features + drop_cols]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_models(pos_weight: float):
    models = [
        (
            "LogReg",
            LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1),
            {"model__C": [0.5, 1.0, 2.0]},
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            {"model__n_estimators": [300], "model__max_depth": [15, None], "model__min_samples_leaf": [1, 3]},
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42),
            {"model__n_estimators": [300], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3]},
        ),
    ]

    if HAS_XGB:
        models.append(
            (
                "XGB",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=pos_weight,
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
                LGBMClassifier(random_state=42, is_unbalance=True),
                {
                    "model__n_estimators": [400],
                    "model__learning_rate": [0.05, 0.1],
                    "model__num_leaves": [63],
                    "model__subsample": [0.8],
                },
            )
        )

    return models


def evaluate_threshold(y_true, proba):
    thresholds = np.linspace(0.1, 0.9, 17)
    best = None
    for t in thresholds:
        preds = (proba >= t).astype(int)
        rec = recall_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        if best is None or rec > best["recall"] or (np.isclose(rec, best["recall"]) and f1 > best["f1"]):
            best = {"threshold": t, "recall": rec, "precision": prec, "f1": f1, "accuracy": acc}
    return best


def plot_curves(y_true, proba, preds, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.colorbar(im)
    fig.savefig(out_dir / "model2_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "model2_roc_curve.png", dpi=300)
    plt.close()

    # Precision-recall
    prec, rec, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "model2_pr_curve.png", dpi=300)
    plt.close()


def main(input_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "is_severe_tomorrow"])
    df = df.sort_values("Date").reset_index(drop=True)

    y = df["is_severe_tomorrow"].astype(int)
    feature_df = df.drop(columns=["is_severe_tomorrow"])

    split_idx = int(len(df) * 0.8)
    X_train, X_test = feature_df.iloc[:split_idx], feature_df.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    preprocessor = build_preprocessor(feature_df)
    models = get_models(pos_weight)
    skf = StratifiedKFold(n_splits=3, shuffle=False)

    results = []
    best = None

    for name, estimator, param_grid in models:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        use_random = name in {"XGB", "LGBM", "RandomForest"}
        search_cls = RandomizedSearchCV if use_random else GridSearchCV
        search_params = {
            "estimator": pipe,
            "cv": skf,
            "scoring": "recall",
            "n_jobs": -1,
            "verbose": 0,
        }
        if param_grid:
            if use_random:
                search_params.update({"param_distributions": param_grid, "n_iter": min(5, sum(len(v) for v in param_grid.values())), "random_state": 42})
            else:
                search_params.update({"param_grid": param_grid})
        else:
            search_params.update({"param_grid": {"model": [estimator]}})

        start = time.time()
        search = search_cls(**search_params)
        search.fit(X_train, y_train)
        duration = time.time() - start

        best_est = search.best_estimator_
        proba_test = best_est.predict_proba(X_test)[:, 1]
        preds_default = (proba_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds_default)
        prec = precision_score(y_test, preds_default, zero_division=0)
        rec = recall_score(y_test, preds_default)
        f1 = f1_score(y_test, preds_default)
        roc = roc_auc_score(y_test, proba_test)

        thresh_info = evaluate_threshold(y_test, proba_test)

        results.append(
            {
                "Model_Name": name,
                "Train_Best_Params": search.best_params_,
                "Test_Accuracy": acc,
                "Test_Precision": prec,
                "Test_Recall": rec,
                "Test_F1": f1,
                "Test_ROC_AUC": roc,
                "CV_Best_Score": search.best_score_,
                "Opt_Threshold": thresh_info["threshold"],
                "Opt_Recall": thresh_info["recall"],
                "Opt_Precision": thresh_info["precision"],
                "Opt_F1": thresh_info["f1"],
                "Training_Time": duration,
                "Best_Estimator": best_est,
                "Proba_Test": proba_test,
            }
        )

        if best is None or thresh_info["recall"] > best["Opt_Recall"] or (
            np.isclose(thresh_info["recall"], best["Opt_Recall"]) and f1 > best["Test_F1"]
        ):
            best = results[-1]

        print(f"Completed {name}: Recall {rec:.3f}, ROC-AUC {roc:.3f}")

    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(["Opt_Recall", "Opt_F1"], ascending=[False, False])
    results_df_sorted.drop(columns=["Best_Estimator", "Proba_Test"], inplace=True)
    results_df_sorted.to_csv(out_dir / "model2_comparison.csv", index=False)

    best_estimator = best["Best_Estimator"]
    best_thresh = best["Opt_Threshold"]
    best_name = best["Model_Name"]
    best_recall = best["Opt_Recall"]
    best_f1 = best["Opt_F1"]

    # Predictions with optimal threshold
    proba_test = best["Proba_Test"]
    preds_opt = (proba_test >= best_thresh).astype(int)

    predictions_df = pd.DataFrame(
        {
            "Date": df.iloc[split_idx:]["Date"],
            "City": df.iloc[split_idx:]["City"],
            "Actual": y_test,
            "Predicted": preds_opt,
            "Probability": proba_test,
        }
    )
    predictions_df.to_csv(out_dir / "model2_predictions.csv", index=False)

    # Plots
    plot_curves(y_test.values, proba_test, preds_opt, out_dir)

    # Clean old model pkls and save best pipeline with metrics in filename
    for old in out_dir.glob("model2_best_*.pkl"):
        try:
            old.unlink()
        except Exception:
            pass
    model_filename = f"model2_best_{best_name}_Recall-{best_recall:.3f}_F1-{best_f1:.3f}.pkl"
    joblib.dump(best_estimator, out_dir / model_filename)

    # Threshold file
    with open(out_dir / "model2_threshold.txt", "w", encoding="utf-8") as f:
        f.write(str(best_thresh))

    # Classification report
    report = classification_report(y_test, preds_opt, digits=3)
    with open(out_dir / "model2_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nBest model:", best_name)
    print("Params:", best.get("Train_Best_Params"))
    print(
        f"Test Accuracy: {accuracy_score(y_test, preds_opt):.3f}, Precision: {precision_score(y_test, preds_opt, zero_division=0):.3f}, Recall: {recall_score(y_test, preds_opt):.3f}, F1: {f1_score(y_test, preds_opt):.3f}, ROC-AUC: {roc_auc_score(y_test, proba_test):.3f}"
    )
    print(f"Optimal threshold: {best_thresh:.2f}")
    print(f"Comparison table saved to {out_dir / 'model2_comparison.csv'}")
    print(f"Predictions saved to {out_dir / 'model2_predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train severe day prediction models")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "model2_severe_day.csv"),
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
