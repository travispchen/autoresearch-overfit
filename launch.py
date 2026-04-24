"""Fan out 15 independent per-task autoresearch loops in parallel.

Each task runs in its own subprocess under a ProcessPoolExecutor. Per-task
loops are fully independent: own `train.py`, own `records.json`, own
`best_val`, own LLM conversation. Stdout from each subprocess is tee'd
into `results/<task>/run.log`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from prepare import TASK_NAMES

REPO = Path(__file__).resolve().parent


def run_task(task: str, n: int, resume: bool) -> tuple[str, int, float]:
    log_dir = REPO / "results" / task
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    cmd = [sys.executable, str(REPO / "run_autoresearch.py"), "--task", task, "--n", str(n)]
    if resume:
        cmd.append("--resume")
    t0 = time.time()
    with log_path.open("a") as f:
        f.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    return task, proc.returncode, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="experiments per task (incl. baseline)")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset (default: all 15)")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks or TASK_NAMES
    print(f"launching {len(tasks)} tasks, n={args.n} each, workers={args.workers}")
    print(f"logs: {REPO / 'results' / '<task>' / 'run.log'}")

    # Avoid oversubscription inside each worker's subprocess.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    t0 = time.time()
    completed = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t, args.n, args.resume): t for t in tasks}
        for f in as_completed(futs):
            task, rc, dt = f.result()
            completed += 1
            status = "OK " if rc == 0 else f"FAIL(rc={rc})"
            if rc != 0:
                failures.append(task)
            print(
                f"[{completed:2d}/{len(tasks):2d}] {status}  {task:40s}  {dt:6.1f}s "
                f"(elapsed {time.time() - t0:.1f}s)"
            )
    print(f"\ntotal elapsed: {time.time() - t0:.1f}s")
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
