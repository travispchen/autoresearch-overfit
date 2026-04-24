"""Per-task val-noise sigma estimate via val-slice bootstrap.

XGBoost with the default baseline config is actually deterministic (no
subsample, no colsample, default tree_method), so seed-to-seed std of
val_err is literally zero. The gate-relevant noise is NOT "seed noise"
but "val-slice noise" — how much would val_err wiggle if our 100-sample
val slice happened to land on slightly different examples?

We estimate this by bootstrap-resampling the val slice: for each task,
train a baseline model once, then compute val_err on B=1000 bootstrap
resamples (with replacement) of the val predictions. The spread is a
principled effective σ for gating.

We also record an "edit-stream" σ: the stddev of val_err across all
non-baseline proposals the LLM generated in Part B. This is an empirical
"how much does val wiggle under real mutations" number. The
effect-size gate uses val-slice σ by default.

Writes `results/_sigma/sigma.json`:

    {
      "per_task": {
        "<task>": {
          "baseline_val_err": float,
          "baseline_test_err": float,
          "bootstrap_val_std": float,
          "bootstrap_test_std": float,
          "edit_stream_val_std": float | null,
        }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES, _load_with_test  # noqa: E402


def bootstrap_std(errors: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    """Bootstrap std of mean(errors) — for classification misclass rate."""
    n = len(errors)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = errors[idx].mean(axis=1)
    return float(means.std(ddof=1))


def edit_stream_std(task: str) -> float | None:
    rec_path = REPO / "results" / task / "records.json"
    if not rec_path.exists():
        return None
    records = json.loads(rec_path.read_text())
    vals = [r["val_err"] for r in records[1:] if r["val_err"] == r["val_err"]]
    if len(vals) < 2:
        return None
    return float(np.array(vals).std(ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(REPO / "results" / "_sigma" / "sigma.json"))
    args = ap.parse_args()

    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    per_task: dict[str, dict] = {}
    print(f"{'task':36s} {'val_err':>8s} {'σ_boot':>8s} {'σ_edit':>8s}")
    for t in TASK_NAMES:
        ds, X_test, y_test = _load_with_test(t)
        model = XGBClassifier(random_state=0, n_jobs=1, eval_metric="mlogloss")
        model.fit(ds.X_train, ds.y_train)
        val_errs = (model.predict(ds.X_val) != ds.y_val).astype(float)
        test_errs = (model.predict(X_test) != y_test).astype(float)
        baseline_val = float(val_errs.mean())
        baseline_test = float(test_errs.mean())
        boot_val = bootstrap_std(val_errs, args.n_boot, rng)
        boot_test = bootstrap_std(test_errs, args.n_boot, rng)
        n_val, n_test = len(val_errs), len(test_errs)

        edit_std = edit_stream_std(t)
        per_task[t] = {
            "baseline_val_err": baseline_val,
            "baseline_test_err": baseline_test,
            "bootstrap_val_std": boot_val,
            "bootstrap_test_std": boot_test,
            "edit_stream_val_std": edit_std,
            "n_val": n_val,
            "n_test": n_test,
        }
        edit_str = f"{edit_std:.5f}" if edit_std is not None else "   n/a"
        print(f"{t:36s} {baseline_val:>8.4f} {boot_val:>8.5f} {edit_str:>8s}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n_boot": args.n_boot, "per_task": per_task}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
