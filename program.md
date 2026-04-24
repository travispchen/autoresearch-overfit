# Per-task autoresearch agent instructions

You are iterating on `train.py` for **one specific** OpenML tabular
classification task. Your goal is to **lower `val_err`** (misclassification
rate on this task's val slice). The starting `train.py` is an untuned
`XGBClassifier` with library defaults.

## Protocol

1. Read the current `train.py`. It defines `train_one(ds) -> predict_fn`
   and calls `evaluate_one(TASK, train_one)` which returns `val_err` for
   the single task identified by `os.environ["TASK"]`. `ds` has
   `X_train, y_train, X_val, y_val, n_classes, n_features`. There is **no**
   test set exposed — `Dataset` does not carry it.
2. Propose ONE concrete code change you believe will reduce `val_err` on
   this specific task. Reason from the code in front of you and your
   knowledge of what helps default XGBoost on this dataset's shape (feature
   count, class count, approximate size).
3. Apply the edit. The harness runs `TASK=<name> uv run python train.py`
   and parses `val_err: <float>` from stdout.
4. If `val_err` is strictly less than the current best, KEEP the edit
   (the harness commits it and updates best). Otherwise DISCARD (the
   harness restores the previous `train.py`).
5. Loop for ~25 experiments.

## What you can mutate

Anything inside `train.py`. The mutation space is wide: `learning_rate`,
`max_depth`, `n_estimators`, `subsample`, `colsample_bytree / bylevel /
bynode`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`,
`tree_method`, `booster` (`gbtree` / `dart` / `gblinear`), `grow_policy`,
`max_leaves`, native categorical handling
(`enable_categorical=True, tree_method="hist"`), class weighting, feature
engineering inside `train_one`, small seed ensembles, etc.

## Hard constraints

- DO NOT modify `prepare.py` or any silent test log.
- DO NOT read `X_test` / `y_test`. `Dataset` does not expose them and you
  must not reconstruct them from OpenML directly.
- DO NOT widen the train split with val data. The val set is the gate.
- Keep `train.py` self-contained, runnable via `TASK=<name> uv run python train.py`.
- Keep the final printed line parseable as `val_err: <float>`.
- Keep the `TASK = os.environ["TASK"]` assignment intact. The harness
  uses the env var to route the call to the right task.

## XGBoost-specific leakage caveat

You MAY pass `eval_set=[(ds.X_val, ds.y_val)]` with `early_stopping_rounds=K`.
This is a legitimate XGBoost pattern. **BUT** it makes the val slice
both the gate *and* the early-stopping oracle, compounding the val leak
this study is measuring. If you use it, **prefix the description with
`[es-on-val]`** so the writeup can segregate those runs.

`eval_set=[(ds.X_train, ds.y_train)]` is fine and does not leak val.

## Style

Small, skimmable code. One focused change per experiment so the val delta
is interpretable. If you suspect the last win was noise (val dropped but
the code change was functionally trivial — e.g. a seed swap, a tiny
numerical tweak), name that explicitly and propose a qualitatively
different change next.
