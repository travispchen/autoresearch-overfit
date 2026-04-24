"""Per-task / per-gate summary table.

Reads `results/<task>/records.json` (= baseline gate) and
`results/<task>/<gate>/records.json` for each mitigation replay, and
emits the aggregate JSON used to produce the §4.1 table in WRITEUP.md.

Writes:
  - results/_summary/gate_comparison_table.json
  - stdout: aggregate row per gate (median v/t, total kept, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES  # noqa: E402


def load_gate(task: str, gate: str | None) -> list[dict]:
    if gate is None:
        p = REPO / "results" / task / "records.json"
    elif gate == "noise_replay":
        p = REPO / "results" / "_noise" / task / "records.json"
    else:
        p = REPO / "results" / task / gate / "records.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def curves(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in records if r["val_err"] == r["val_err"]]
    val_c = np.zeros(len(rows))
    test_c = np.zeros(len(rows))
    best_val = float("inf")
    best_test = float("nan")
    for i, r in enumerate(rows):
        # replayed records carry best_*_after; baseline records carry
        # per-experiment val/test and we need to walk keeps.
        if "best_val_after" in r:
            best_val = r["best_val_after"]
            best_test = r["best_test_after"]
        else:
            if r["val_err"] < best_val:
                best_val = r["val_err"]
                best_test = r["test_err"]
        val_c[i] = best_val
        test_c[i] = best_test
    return val_c, test_c


def summarize(records: list[dict]) -> dict:
    val_c, test_c = curves(records)
    if len(val_c) < 2:
        return {}
    vd = float(val_c[0] - val_c[-1])
    td = float(test_c[0] - test_c[-1])
    ratio = vd / td if abs(td) > 1e-9 else float("inf")
    kept = sum(1 for r in records if r["status"] == "keep")
    return {
        "n": len(val_c),
        "kept": kept,
        "baseline_val": float(val_c[0]),
        "final_val": float(val_c[-1]),
        "val_drop": vd,
        "baseline_test": float(test_c[0]),
        "final_test": float(test_c[-1]),
        "test_drop": td,
        "val_test_ratio": ratio,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="*", default=None)
    ap.add_argument(
        "--out-table", default=str(REPO / "results" / "_summary" / "gate_comparison_table.json")
    )
    args = ap.parse_args()

    # Discover gates if not specified: any subdir of results/<task>/ that holds records.json.
    if args.gates is None:
        gate_set: set[str] = set()
        for t in TASK_NAMES:
            base = REPO / "results" / t
            if not base.exists():
                continue
            for sub in base.iterdir():
                if sub.is_dir() and (sub / "records.json").exists():
                    gate_set.add(sub.name)
        noise_dir = REPO / "results" / "_noise"
        if noise_dir.exists() and any(
            (noise_dir / t / "records.json").exists() for t in TASK_NAMES
        ):
            gate_set.add("noise_replay")
        args.gates = sorted(gate_set)
    gates = ["baseline"] + list(args.gates)

    table: dict[str, dict[str, dict]] = {g: {} for g in gates}
    for task in TASK_NAMES:
        for g in gates:
            gate_key = None if g == "baseline" else g
            recs = load_gate(task, gate_key)
            if not recs:
                continue
            table[g][task] = summarize(recs)

    # Aggregate table
    agg = {}
    for g in gates:
        ratios = []
        kept = []
        vds = []
        tds = []
        for t in TASK_NAMES:
            if t not in table[g]:
                continue
            row = table[g][t]
            if np.isfinite(row["val_test_ratio"]):
                ratios.append(row["val_test_ratio"])
            kept.append(row["kept"])
            vds.append(row["val_drop"])
            tds.append(row["test_drop"])
        agg[g] = {
            "n_tasks": len(kept),
            "median_val_test_ratio": float(np.median(ratios)) if ratios else float("nan"),
            "mean_val_drop": float(np.mean(vds)) if vds else float("nan"),
            "mean_test_drop": float(np.mean(tds)) if tds else float("nan"),
            "total_kept": int(sum(kept)),
            "n_overfit_r3": int(sum(1 for r in ratios if r >= 3.0)),
            "n_honest_r1p5": int(sum(1 for r in ratios if r <= 1.5)),
        }

    out_table = {"per_task": table, "aggregate": agg, "gates": gates}
    Path(args.out_table).write_text(json.dumps(out_table, indent=2))
    print(f"wrote {args.out_table}")

    print(
        f"\n{'gate':28s} {'n':>3s} {'med_v/t':>8s} {'μΔval':>8s} {'μΔtest':>8s} "
        f"{'kept':>5s} {'r≥3':>4s} {'r≤1.5':>5s}"
    )
    for g in gates:
        a = agg[g]
        r = a["median_val_test_ratio"]
        rstr = f"{r:7.2f}x" if np.isfinite(r) else "   inf"
        print(
            f"{g:28s} {a['n_tasks']:3d} {rstr} {a['mean_val_drop']:>+8.4f} "
            f"{a['mean_test_drop']:>+8.4f} {a['total_kept']:5d} "
            f"{a['n_overfit_r3']:4d} {a['n_honest_r1p5']:5d}"
        )


if __name__ == "__main__":
    main()
