"""Re-execute every unique train.py from each task's proposal stream so the
silent log picks up per-sample val predictions.

Existing silent log entries written before the per-sample capture landed have
no `val_per_sample_b64` field. We replay each unique (task, train_py_hash)
pair once — writes a fresh silent-log entry with the per-sample array, and
keeps records.json untouched.

Runs one task at a time in parallel workers (ProcessPoolExecutor). Uses the
harness's own run_train subprocess so the exact same code path that
populated the original results is what re-populates the log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES, silent_log_path  # noqa: E402

SUBPROCESS_TIMEOUT = 600


def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def silent_log_has_per_sample(task: str) -> set[str]:
    """Return set of train_py_hash values already captured with per-sample data."""
    path = silent_log_path(task)
    if not path.exists():
        return set()
    hashes: set[str] = set()
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if "val_per_sample_b64" in rec and "train_py_hash" in rec:
                hashes.add(rec["train_py_hash"])
    return hashes


def run_train(task: str, task_dir: Path) -> None:
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "--project", str(REPO), "python", str(task_dir / "train.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        env={**__import__("os").environ, "TASK": task, "PYTHONPATH": str(REPO)},
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"train.py exit={proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )


def run_task(task: str) -> tuple[str, int, int, float]:
    t0 = time.time()
    task_dir = REPO / "results" / task
    proposals_path = task_dir / "proposals.jsonl"
    records = json.loads((task_dir / "records.json").read_text())
    # collect unique train.py sources from proposals.jsonl (keep + discard, excluding errors)
    hash_to_src: dict[str, str] = {}
    if proposals_path.exists():
        with proposals_path.open() as f:
            for line in f:
                r = json.loads(line)
                src = r["new_train_py"]
                h = hash_bytes(src.encode("utf-8"))
                hash_to_src[h] = src
    # Record index 0 is the initial train.py before any proposal — make sure
    # the baseline source is in the hash map so we can re-run it too.
    from run_autoresearch import BASELINE_SRC  # noqa: E402

    hash_to_src[hash_bytes(BASELINE_SRC.encode("utf-8"))] = BASELINE_SRC

    # Only re-run hashes that actually appear in records with non-nan val_err
    needed: set[str] = set()
    for rec in records:
        if rec["val_err"] == rec["val_err"] and "train_py_hash" in rec:
            needed.add(rec["train_py_hash"])
    already = silent_log_has_per_sample(task)
    todo = sorted(needed - already)
    if not todo:
        return task, 0, 0, time.time() - t0

    n_ok = 0
    for h in todo:
        assert h in hash_to_src, f"{task}: no source for hash {h}"
        (task_dir / "train.py").write_text(hash_to_src[h])
        run_train(task, task_dir)
        n_ok += 1
    return task, n_ok, len(todo), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = args.tasks or TASK_NAMES
    print(f"populating per-sample val for {len(tasks)} tasks, workers={args.workers}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for f in as_completed(futs):
            task, ok, total, dt = f.result()
            print(f"  {task:40s} {ok}/{total} reruns in {dt:.1f}s")
    print(f"total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
