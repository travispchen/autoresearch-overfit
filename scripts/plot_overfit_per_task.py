"""3×5 per-task summary grid + per-task summary table.

Reads `results/<task>/records.json` for each task (optionally under a
mitigation subdir, e.g. `results/<task>/<gate>/records.json`) and writes:

  - {out_dir}/per_task_grid.png   : 3×5 grid, one subplot per task
  - {out_dir}/summary.json        : per-task summary rows
  - stdout                        : pretty-printed table, aggregate row

Summary columns: baseline_val, final_val, val_drop, baseline_test,
final_test, test_drop, val:test ratio, kept count, n_experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES  # noqa: E402


def load_records(task: str, subdir: str | None) -> list[dict]:
    base = REPO / "results" / task
    path = base / subdir / "records.json" if subdir else base / "records.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def trajectory(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [r for r in records if r["val_err"] == r["val_err"]]
    best_val = float("inf")
    test_at_best = float("nan")
    val_curve = np.zeros(len(rows))
    test_curve = np.zeros(len(rows))
    raw_val = np.zeros(len(rows))
    raw_test = np.zeros(len(rows))
    for i, r in enumerate(rows):
        raw_val[i] = r["val_err"]
        raw_test[i] = r["test_err"]
        if r["val_err"] < best_val:
            best_val = r["val_err"]
            test_at_best = r["test_err"]
        val_curve[i] = best_val
        test_curve[i] = test_at_best
    return val_curve, test_curve, raw_val, raw_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default=None, help="mitigation subdir under results/<task>/")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    tag_path = REPO / "results" / "_summary" / args.tag
    out_dir = Path(args.out_dir) if args.out_dir else tag_path
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_out = out_dir / "per_task_grid.png"
    summary_out = out_dir / "summary.json"

    loaded: list[tuple[str, list[dict]]] = []
    for t in TASK_NAMES:
        recs = load_records(t, args.subdir)
        if recs:
            loaded.append((t, recs))
    assert loaded, "no results/<task>/records.json found"

    fig, axes = plt.subplots(4, 4, figsize=(20, 12), sharex=False, sharey=False)
    axes = axes.flatten()
    for k in range(len(loaded), len(axes)):
        axes[k].set_visible(False)
    summaries = []
    for k, (task, recs) in enumerate(loaded):
        ax = axes[k]
        val_c, test_c, raw_v, raw_t = trajectory(recs)
        xs = np.arange(len(val_c))
        ax.scatter(xs, raw_v, s=6, alpha=0.3, color="tab:blue")
        ax.scatter(xs, raw_t, s=6, alpha=0.3, color="tab:orange")
        ax.plot(xs, val_c, color="tab:blue", lw=1.7, label="val best-so-far")
        ax.plot(xs, test_c, color="tab:orange", lw=1.7, label="test @ val-best")
        val_drop = float(val_c[0] - val_c[-1])
        test_drop = float(test_c[0] - test_c[-1])
        ratio = val_drop / test_drop if abs(test_drop) > 1e-9 else float("inf")
        kept = sum(1 for r in recs if r["status"] == "keep")
        ax.set_title(
            f"{task}\nΔv={val_drop:+.3f} Δt={test_drop:+.3f} r={ratio:.1f}× kept={kept}/{len(recs)}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        summaries.append(
            {
                "task": task,
                "n_experiments": len(recs),
                "kept": kept,
                "baseline_val": float(val_c[0]),
                "final_val": float(val_c[-1]),
                "val_drop": val_drop,
                "baseline_test": float(test_c[0]),
                "final_test": float(test_c[-1]),
                "test_drop": test_drop,
                "val_test_ratio": ratio,
            }
        )
    for j in range(len(loaded), len(axes)):
        axes[j].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Per-task val-gated autoresearch — {args.tag}", fontsize=13)
    fig.supxlabel("experiment index")
    fig.supylabel("misclassification error")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(grid_out, dpi=140)
    plt.close(fig)
    print(f"wrote {grid_out}")

    summaries.sort(key=lambda s: -s["val_test_ratio"] if np.isfinite(s["val_test_ratio"]) else -1e9)
    summary_out.write_text(json.dumps(summaries, indent=2))
    print(f"wrote {summary_out}")

    print(f"\n{'task':36s} {'n':>3s} {'kept':>5s} {'Δval':>8s} {'Δtest':>8s} {'v/t':>8s}")
    for s in summaries:
        r = s["val_test_ratio"]
        rstr = f"{r:>7.2f}x" if np.isfinite(r) else "   inf"
        print(
            f"{s['task']:36s} {s['n_experiments']:3d} {s['kept']:5d} "
            f"{s['val_drop']:>8.4f} {s['test_drop']:>8.4f} {rstr}"
        )

    vd = np.array([s["val_drop"] for s in summaries])
    td = np.array([s["test_drop"] for s in summaries])
    ratios = np.array([s["val_test_ratio"] for s in summaries if np.isfinite(s["val_test_ratio"])])
    n_overfit = sum(1 for r in ratios if r >= 3.0)
    n_honest = sum(1 for r in ratios if r <= 1.5)
    n_middle = len(ratios) - n_overfit - n_honest
    print(f"\nmean Δval  = {vd.mean():+.4f}")
    print(f"mean Δtest = {td.mean():+.4f}")
    print(f"median v/t = {np.median(ratios):.2f}x" if len(ratios) else "median v/t = n/a")
    print(
        f"overfit (r>=3):   {n_overfit}/{len(loaded)}\n"
        f"honest  (r<=1.5): {n_honest}/{len(loaded)}\n"
        f"middle:           {n_middle}/{len(loaded)}"
    )


if __name__ == "__main__":
    main()
