"""Statistical significance of the overfit findings.

For each gate, compute per-task (Δval, Δtest, gap) where
  Δval  = val_err_at_exp_0 − val_err_at_val_best_so_far_exp_last
  Δtest = test_err_at_exp_0 − test_err_at_val_best_so_far_exp_last
  gap   = Δval − Δtest      (positive = val-gate overfit)

Then:
  - One-sample Wilcoxon signed-rank on baseline Δtest vs 0 (is there overfit?)
  - Paired Wilcoxon on (gate_Δtest − baseline_Δtest)     (does gate help test?)
  - Paired Wilcoxon on (baseline_gap − gate_gap)         (does gate close gap?)
  - Bootstrap 95% CI for mean Δtest and mean gap, per gate.

Writes results/_summary/significance.json and prints a summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES  # noqa: E402

GATES = [
    "baseline",
    "effect_size_k1.0",
    "rotating_val",
    "thresholdout_b10",
    "topk_confirm_k3",
    "reflection",
    "noise_replay",
]


def load(task: str, gate: str) -> list[dict]:
    if gate == "baseline":
        p = REPO / "results" / task / "records.json"
    elif gate == "noise_replay":
        p = REPO / "results" / "_noise" / task / "records.json"
    else:
        p = REPO / "results" / task / gate / "records.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def per_task_stats(gate: str) -> dict[str, tuple[float, float, float]]:
    """Map task -> (Δval, Δtest, gap) for this gate."""
    out: dict[str, tuple[float, float, float]] = {}
    for t in TASK_NAMES:
        recs = load(t, gate)
        if not recs:
            continue
        rows = [r for r in recs if r["val_err"] == r["val_err"]]
        if len(rows) < 2:
            continue
        best_val = float("inf")
        best_test = float("nan")
        for r in rows:
            if "best_val_after" in r:
                best_val = r["best_val_after"]
                best_test = r["best_test_after"]
            elif r["val_err"] < best_val:
                best_val = r["val_err"]
                best_test = r["test_err"]
        v0 = rows[0]["val_err"]
        t0 = rows[0]["test_err"]
        dv = v0 - best_val
        dt = t0 - best_test
        out[t] = (dv, dt, dv - dt)
    return out


def bootstrap_mean_ci(xs: np.ndarray, n_boot: int = 20000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
    means = xs[idx].mean(axis=1)
    return float(xs.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    per_gate = {g: per_task_stats(g) for g in GATES}
    base = per_gate["baseline"]
    base_tasks = sorted(base.keys())

    # 1) Does baseline overfit? (test: Δtest > 0 means test improved; we claim baseline's
    # Δtest is basically 0 or negative while Δval > 0)
    b_dval = np.array([base[t][0] for t in base_tasks])
    b_dtest = np.array([base[t][1] for t in base_tasks])
    b_gap = np.array([base[t][2] for t in base_tasks])

    print("=" * 100)
    print("Part 1: Is there overfit in the baseline?")
    print("=" * 100)
    # Wilcoxon requires at least one non-zero value; assert rather than silently
    # returning None (which would then crash on `.pvalue` below).
    assert (b_dval != 0).any(), "baseline Δval is all zeros — no signal to test"
    assert (b_gap != 0).any(), "baseline gap is all zeros — no signal to test"
    w_dval = stats.wilcoxon(b_dval, alternative="greater")
    w_gap = stats.wilcoxon(b_gap, alternative="greater")
    # Paired test: is Δval > Δtest per task?
    w_paired = stats.wilcoxon(b_dval, b_dtest, alternative="greater")
    m_dval, lo_dval, hi_dval = bootstrap_mean_ci(b_dval)
    m_dtest, lo_dtest, hi_dtest = bootstrap_mean_ci(b_dtest)
    m_gap, lo_gap, hi_gap = bootstrap_mean_ci(b_gap)
    print(f"  Baseline Δval   mean={m_dval:+.4f}   95% CI [{lo_dval:+.4f}, {hi_dval:+.4f}]")
    print(f"  Baseline Δtest  mean={m_dtest:+.4f}   95% CI [{lo_dtest:+.4f}, {hi_dtest:+.4f}]")
    print(f"  Baseline gap    mean={m_gap:+.4f}   95% CI [{lo_gap:+.4f}, {hi_gap:+.4f}]")
    print(f"  One-sided Wilcoxon: Δval > 0           p = {w_dval.pvalue:.4f}")
    print(f"  One-sided Wilcoxon: gap > 0            p = {w_gap.pvalue:.4f}")
    print(f"  Paired Wilcoxon:   Δval_i > Δtest_i   p = {w_paired.pvalue:.4f}")
    print()

    # 2) For each gate, paired Wilcoxon vs baseline on:
    #    (a) is gate_gap < baseline_gap?     (does gate close overfit gap?)
    #    (b) is gate_dtest > baseline_dtest? (does gate improve test outcome?)
    print("=" * 100)
    print(
        f"Part 2: Per-gate paired tests vs baseline (n={len(base_tasks)} tasks, "
        "Wilcoxon signed-rank, one-sided)"
    )
    print("=" * 100)
    header = f"{'gate':28s}  {'mean Δtest':>12s}  {'95% CI':>20s}  {'mean gap':>10s}  "
    header += f"{'gap < base':>11s}  {'Δtest > base':>13s}"
    print(header)
    print("-" * len(header))

    out_rows = []
    for g in GATES:
        gate_stats = per_gate[g]
        shared = [t for t in base_tasks if t in gate_stats]
        if len(shared) < 2:
            continue
        g_dtest = np.array([gate_stats[t][1] for t in shared])
        g_gap = np.array([gate_stats[t][2] for t in shared])
        b_dtest_s = np.array([base[t][1] for t in shared])
        b_gap_s = np.array([base[t][2] for t in shared])

        m_gt, lo_gt, hi_gt = bootstrap_mean_ci(g_dtest)
        m_gg, _, _ = bootstrap_mean_ci(g_gap)

        if g == "baseline":
            p_gap = 1.0
            p_test = 1.0
        else:
            # is gate_gap < base_gap ?
            diffs_gap = b_gap_s - g_gap  # positive = gate better
            diffs_test = g_dtest - b_dtest_s  # positive = gate better
            if (diffs_gap != 0).any():
                p_gap = stats.wilcoxon(diffs_gap, alternative="greater").pvalue
            else:
                p_gap = float("nan")
            if (diffs_test != 0).any():
                p_test = stats.wilcoxon(diffs_test, alternative="greater").pvalue
            else:
                p_test = float("nan")

        out_rows.append(
            dict(
                gate=g,
                n=len(shared),
                mean_dtest=float(m_gt),
                ci_dtest_lo=float(lo_gt),
                ci_dtest_hi=float(hi_gt),
                mean_gap=float(m_gg),
                p_gap_lt_baseline=float(p_gap),
                p_dtest_gt_baseline=float(p_test),
            )
        )
        print(
            f"{g:28s}  {m_gt:+11.4f}  [{lo_gt:+.4f},{hi_gt:+.4f}]  "
            f"{m_gg:+10.4f}  {p_gap:>11.4f}  {p_test:>13.4f}"
        )

    # 3) Reflection specifically: does it hurt vs baseline? Two-sided.
    print()
    print("=" * 100)
    print("Part 3: Reflection — does it hurt? (two-sided paired Wilcoxon on Δtest)")
    print("=" * 100)
    refl = per_gate["reflection"]
    shared = [t for t in base_tasks if t in refl]
    r_dtest = np.array([refl[t][1] for t in shared])
    b_dtest_s = np.array([base[t][1] for t in shared])
    diffs = r_dtest - b_dtest_s
    if (diffs != 0).any():
        w = stats.wilcoxon(diffs, alternative="two-sided")
        w_worse = stats.wilcoxon(diffs, alternative="less")
        print(
            f"  mean(reflection_Δtest − baseline_Δtest) = {diffs.mean():+.4f} "
            f"across {len(shared)} tasks"
        )
        print(f"  two-sided p = {w.pvalue:.4f}")
        print(f"  one-sided p (reflection_Δtest < baseline_Δtest) = {w_worse.pvalue:.4f}")

    # 4) Noise replay: does its Δval explain most of the baseline's Δval?
    print()
    print("=" * 100)
    print("Part 4: Noise_replay — is its Δval indistinguishable from baseline's Δval?")
    print("=" * 100)
    nr = per_gate["noise_replay"]
    shared = [t for t in base_tasks if t in nr]
    n_dval = np.array([nr[t][0] for t in shared])
    b_dval_s = np.array([base[t][0] for t in shared])
    diffs = b_dval_s - n_dval
    w = stats.wilcoxon(diffs, alternative="two-sided")
    print(f"  baseline mean Δval = {b_dval_s.mean():+.4f}")
    print(f"  noise    mean Δval = {n_dval.mean():+.4f}")
    print(f"  ratio noise/baseline = {n_dval.mean() / b_dval_s.mean():.2f}")
    print(f"  paired Wilcoxon (two-sided, baseline-Δval vs noise-Δval): p = {w.pvalue:.4f}")

    # Save JSON
    out_path = REPO / "results" / "_summary" / "significance.json"
    payload = {
        "baseline": {
            "dval_mean": m_dval,
            "dval_ci95": [lo_dval, hi_dval],
            "dtest_mean": m_dtest,
            "dtest_ci95": [lo_dtest, hi_dtest],
            "gap_mean": m_gap,
            "gap_ci95": [lo_gap, hi_gap],
            "p_dval_gt_0": float(w_dval.pvalue),
            "p_gap_gt_0": float(w_gap.pvalue),
            "p_dval_gt_dtest_paired": float(w_paired.pvalue),
        },
        "per_gate_vs_baseline": out_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
