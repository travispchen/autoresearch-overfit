"""Bonus: pure-noise mutation replay against the XGBoost baseline, per-task.

Replaces the LLM proposer with a deterministic script that emits:
  - the starting baseline (i=0)
  - a short prefix of REAL XGBoost mutations (give val some real headroom)
  - a long tail of pure seed swaps (random_state=N), which are zero-
    information-content noise

Pushes that stream through the standard val-gate per task and shows that
val still descends more than test. Isolates the adaptive-holdout-reuse
effect from LLM proposal quality.

Not the main study; bonus/lower-bound proof. Writes
`results/_noise/<task>/records.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES, silent_log_path  # noqa: E402
from run_autoresearch import BASELINE_SRC  # noqa: E402

_VAL_ERR_RE = re.compile(r"^val_err:\s*([0-9.]+)\s*$", re.MULTILINE)


# Each REAL_MUTATION is a STANDALONE replacement of the XGBClassifier call, not
# cumulative. This keeps each mutation applicable regardless of what was kept
# before. Seed swaps layer on top of whichever variant is current.
_BASE_CALL = (
    "model = XGBClassifier(\n        random_state=0,\n        n_jobs=1,\n        "
    'eval_metric="mlogloss",\n    )'
)

REAL_MUTATIONS = [
    ("lr=0.1, n_estimators=300", "learning_rate=0.1, n_estimators=300, "),
    ("lr=0.05, n_estimators=500", "learning_rate=0.05, n_estimators=500, "),
    ("max_depth=4, n_estimators=300", "learning_rate=0.1, n_estimators=300, max_depth=4, "),
    (
        "subsample=0.8, colsample_bytree=0.8",
        "learning_rate=0.1, n_estimators=300, subsample=0.8, colsample_bytree=0.8, ",
    ),
    (
        "min_child_weight=3",
        "learning_rate=0.1, n_estimators=300, min_child_weight=3, ",
    ),
]


def apply_real(src: str, idx: int) -> tuple[str, str]:
    desc, extra = REAL_MUTATIONS[idx]
    replacement = _BASE_CALL.replace(
        'eval_metric="mlogloss",\n    )',
        f'eval_metric="mlogloss",\n        {extra.strip()}\n    )',
    )
    # Find the CURRENT XGBClassifier call in src and replace it wholesale.
    call_re = re.compile(r"model = XGBClassifier\([^)]*\)", re.DOTALL)
    m = call_re.search(src)
    assert m is not None, "no XGBClassifier(...) found in src"
    new = src[: m.start()] + replacement + src[m.end() :]
    return new, desc


def apply_seed(src: str, seed: int) -> tuple[str, str]:
    new, n = re.subn(r"random_state=\d+", f"random_state={seed}", src)
    assert n >= 1
    return new, f"seed-swap: random_state={seed}"


def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def run_train(task: str, train_path: Path) -> float:
    env = os.environ.copy()
    env["TASK"] = task
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env.get("PYTHONPATH", "")).rstrip(os.pathsep)
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "--project", str(REPO), "python", str(train_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, f"{task} failed:\n{proc.stderr[-800:]}"
    m = _VAL_ERR_RE.search(proc.stdout)
    assert m is not None
    return float(m.group(1))


def _tail(path: Path) -> dict:
    with path.open() as f:
        last = None
        for line in f:
            if line.strip():
                last = line
    assert last is not None
    return json.loads(last)


def run_one_task(task: str, n_experiments: int) -> tuple[str, int]:
    out_dir = REPO / "results" / "_noise" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.py"
    train_path.write_text(BASELINE_SRC)
    cache_path = out_dir / "eval_cache.json"
    cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    log_path = silent_log_path(task)

    def eval_src(src: str) -> dict:
        h = content_hash(src)
        if h in cache:
            return cache[h]
        train_path.write_text(src)
        _ = run_train(task, train_path)
        rec = _tail(log_path)
        cache[h] = {
            "train_err": rec["train_err"],
            "val_err": rec["val_err"],
            "test_err": rec["test_err"],
            "train_py_hash": h,
        }
        cache_path.write_text(json.dumps(cache, indent=2))
        return cache[h]

    records: list[dict] = []
    current = train_path.read_text()
    ev0 = eval_src(current)
    best_val = ev0["val_err"]
    records.append(
        {
            "i": 0,
            "desc": "baseline",
            "train_err": ev0["train_err"],
            "val_err": ev0["val_err"],
            "test_err": ev0["test_err"],
            "status": "keep",
            "train_py_hash": ev0["train_py_hash"],
        }
    )

    # First apply real mutations sequentially on top of the current best.
    for idx in range(min(len(REAL_MUTATIONS), n_experiments - 1)):
        mutated, desc = apply_real(current, idx)
        ev = eval_src(mutated)
        status = "keep" if ev["val_err"] < best_val else "discard"
        if status == "keep":
            best_val = ev["val_err"]
            current = mutated  # accept: build future seed swaps on top of this
        records.append(
            {
                "i": len(records),
                "desc": desc,
                "train_err": ev["train_err"],
                "val_err": ev["val_err"],
                "test_err": ev["test_err"],
                "status": status,
                "train_py_hash": ev["train_py_hash"],
            }
        )

    # Then a long tail of seed swaps.
    seeds_tried = 0
    seed = 1
    while len(records) < n_experiments:
        swapped, desc = apply_seed(current, seed)
        seeds_tried += 1
        seed += 1
        ev = eval_src(swapped)
        status = "keep" if ev["val_err"] < best_val else "discard"
        if status == "keep":
            best_val = ev["val_err"]
            current = swapped
        records.append(
            {
                "i": len(records),
                "desc": desc,
                "train_err": ev["train_err"],
                "val_err": ev["val_err"],
                "test_err": ev["test_err"],
                "status": status,
                "train_py_hash": ev["train_py_hash"],
            }
        )
    (out_dir / "records.json").write_text(json.dumps(records, indent=2))
    return task, seeds_tried


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()

    tasks = args.tasks or TASK_NAMES
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one_task, t, args.n) for t in tasks]
        for f in as_completed(futs):
            task, seeds = f.result()
            done += 1
            print(
                f"[{done:2d}/{len(tasks):2d}] {task:40s} seeds={seeds} "
                f"elapsed {time.time() - t0:.1f}s"
            )


if __name__ == "__main__":
    main()
