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

TASK = os.environ["TASK"]


def train_one(ds: Dataset):
    assert ds.task_type == "classification"
    model = XGBClassifier(
        random_state=0,
        n_jobs=1,
        eval_metric="mlogloss",
        min_child_weight=3,
        gamma=0.1,
    )
    model.fit(ds.X_train, ds.y_train)

    def predict(X_np):
        return model.predict(X_np)

    return predict


if __name__ == "__main__":
    out = evaluate_one(TASK, train_one)
    print(f"val_err: {out['val_err']:.4f}")