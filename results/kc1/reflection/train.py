"""Per-task baseline train.py (agent-editable).

Every per-task autoresearch loop gets its own copy of this file at
`results/<task>/train.py` and mutates that copy. The top-level `train.py`
is just the starting template; `run_autoresearch.py` copies it into the
per-task results dir at the start of each run.

TASK is injected via env var by the driver, so the same file works for all
15 tasks without modification.
"""

# prepare must be imported first so it sets OMP_NUM_THREADS=1 before xgboost's
# OpenMP pool is initialized. Do not reorder.
import os  # noqa: I001
from prepare import Dataset, evaluate_one
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

TASK = os.environ["TASK"]


def train_one(ds: Dataset):
    assert ds.task_type == "classification"
    
    # Create polynomial features (degree=2, interaction only)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(ds.X_train)
    X_val_poly = poly.transform(ds.X_val)
    
    model = XGBClassifier(
        random_state=0,
        n_jobs=1,
        eval_metric="mlogloss",
        learning_rate=0.03,
        n_estimators=400,
        max_depth=4,
        colsample_bytree=0.7,
        colsample_bynode=0.8,
        subsample=0.85,
        reg_alpha=0.1,
        reg_lambda=0.3,
    )
    model.fit(X_train_poly, ds.y_train)

    def predict(X_np):
        X_poly = poly.transform(X_np)
        return model.predict(X_poly)

    return predict


if __name__ == "__main__":
    out = evaluate_one(TASK, train_one)
    print(f"val_err: {out['val_err']:.4f}")