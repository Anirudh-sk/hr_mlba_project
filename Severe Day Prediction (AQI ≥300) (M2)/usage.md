# Model 2 Usage (Severe Day Prediction)

Best model: `model2_best_RandomForest_Recall-0.987_F1-0.678.pkl` (pipeline with preprocessing + classifier).
Optimal decision threshold: see `model2_threshold.txt` (default 0.20 from training).

## Predicting severe day
```python
import joblib
import pandas as pd
from pathlib import Path

root = Path(__file__).parent
model = joblib.load(root / "model2_best_RandomForest_Recall-0.987_F1-0.678.pkl")
threshold = float((root / "model2_threshold.txt").read_text().strip())

# Load or assemble data with the same feature columns as model2_severe_day.csv
X = pd.read_csv(root / "model2_severe_day.csv")
X = X.drop(columns=["is_severe_tomorrow"], errors="ignore")

proba = model.predict_proba(X)[:, 1]
preds = (proba >= threshold).astype(int)
```

## Expected inputs
- Columns matching the prepared dataset (lags, rolling stats, temporal flags, change features, severe indicators). `City` can be new; it is one-hot encoded. `Date` is ignored by the pipeline.
- No manual scaling/encoding needed; the pipeline handles preprocessing and imputation.

## Outputs
- `proba`: probability of a severe day tomorrow (AQI ≥ 300).
- `preds`: binary 0/1 using the optimal threshold.
