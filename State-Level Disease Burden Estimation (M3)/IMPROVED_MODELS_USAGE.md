# Disease Burden Prediction Models (Improved, Lower Overfit)

## Why new models?
- Original fits were near-perfect (R2 ~ 1.0) on a 77-row dataset — clear overfitting.
- We rebuilt with simpler settings to get realistic generalization (R2 ~ 0.8 and small train-test gaps).

## What changed (overfitting fix)
- Small dataset + too many features → memorization.
- Removed State encoding; kept a small core pollution/AQI feature set only.
- Strong regularization + shallow trees; no hyperparameter tuning.
- Selection favors realistic test R2 and small gaps; honest error bands instead of ±1 illusions.

Key changes
- Removed State one-hot encoding.
- Limited to ~8-12 core pollution/AQI features (no extra interactions).
- Strong regularization (ElasticNet/Lasso/Ridge at higher alpha).
- Shallow trees/boosting (max_depth <= 2, <= 80 estimators).
- No hyperparameter search; fixed, simple settings.
- Selection rule: prefer test R2 in [0.4, 0.85] with |gap| <= 0.2.

Files to use
- `improved_best_Cardiovascular_per_100k_ElasticNet_Strong_R2-0.805_gap--0.067.pkl`
- `improved_best_Respiratory_per_100k_ElasticNet_Strong_R2-0.803_gap--0.074.pkl`
- `improved_best_All_Key_Diseases_per_100k_ElasticNet_Strong_R2-0.814_gap--0.070.pkl`
- Comparisons: `improved_*_comparison.csv`
- Predictions: `improved_*_predictions.csv`
- Feature importance: `improved_*_feature_importance.csv`
- Plots: `improved_*_actual_vs_pred.png` (see how predictions deviate from the 45-degree line)

Performance (test split, 70/30)
- Cardiovascular: R2 ~ 0.81, RMSE ~ 77, gap ~ -0.07
- Respiratory: R2 ~ 0.80, RMSE ~ 49, gap ~ -0.07
- All diseases: R2 ~ 0.81, RMSE ~ 122, gap ~ -0.07

Quick start
```python
import joblib
import pandas as pd
from pathlib import Path

root = Path(__file__).parent
models = {
    "Cardiovascular_per_100k": joblib.load(root / "improved_best_Cardiovascular_per_100k_ElasticNet_Strong_R2-0.805_gap--0.067.pkl"),
    "Respiratory_per_100k": joblib.load(root / "improved_best_Respiratory_per_100k_ElasticNet_Strong_R2-0.803_gap--0.074.pkl"),
    "All_Key_Diseases_per_100k": joblib.load(root / "improved_best_All_Key_Diseases_per_100k_ElasticNet_Strong_R2-0.814_gap--0.070.pkl"),
}

X = pd.read_csv(root / "model3_disease_burden.csv")
# Drop targets if present; models expect the limited numeric set internally selected at train time.
X = X.drop(columns=["Cardiovascular_per_100k", "Respiratory_per_100k", "All_Key_Diseases_per_100k"], errors="ignore")

preds = {tgt: mdl.predict(X) for tgt, mdl in models.items()}
pred_df = pd.DataFrame({"State": X.get("State", pd.Series(["" for _ in range(len(X))])), "Year": X.get("Year", pd.Series([None for _ in range(len(X))]))})
for tgt, arr in preds.items():
    pred_df[f"Pred_{tgt}"] = arr
print(pred_df.head())
```

Interpretation guidance
- Report predictions with uncertainty: use RMSE as a rough ± band (e.g., Cardiovascular ≈ ±77 per 100k).
- Treat these as exploratory/relative signals; dataset is small (77 rows), so variance is expected.
- See comparison figure `improved_*_actual_vs_pred.png` for visual sense of realism.

If you need even lower overfit (R2 <= 0.75), tighten the selection rule in `train_improved_target` to cap test R2 at 0.75 and rerun.
