"""Compare the LLM-as-judge complexity gate against baseline + effect_size_k1.0.

Reuses the exact per-task (Δval, Δtest, gap) computation and paired Wilcoxon
tests from stats_significance.py, so the judge gate is scored apples-to-apples
with the published gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from stats_significance import bootstrap_mean_ci, per_task_stats  # noqa: E402

GATES = [
    "baseline",
    "effect_size_k0.5",
    "effect_size_k1.0",
    "effect_size_k2.0",
    "judge_k1.0",
    "judge_k2.0",
    "judge_k3.0",
    "judge_noise",
    "judge_hybrid",
]


def main() -> None:
    per_gate = {g: per_task_stats(g) for g in GATES}
    base = per_gate["baseline"]
    tasks = sorted(base.keys())

    print(
        f"{'gate':18s} {'μΔval':>8s} {'μΔtest':>8s} {'μgap':>8s} {'kept':>5s} "
        f"{'gap<base p':>11s} {'Δtest>base p':>13s}"
    )
    print("-" * 78)
    for g in GATES:
        gs = per_gate[g]
        shared = [t for t in tasks if t in gs]
        dval = np.array([gs[t][0] for t in shared])
        dtest = np.array([gs[t][1] for t in shared])
        gap = np.array([gs[t][2] for t in shared])
        m_dval = dval.mean()
        m_dtest, _, _ = bootstrap_mean_ci(dtest)
        m_gap = gap.mean()
        # mean kept count across tasks for this gate
        kept = []
        for t in shared:
            p = (
                (REPO / "results" / t / "records.json")
                if g == "baseline"
                else (REPO / "results" / t / g / "records.json")
            )
            recs = json.loads(p.read_text())
            kept.append(sum(1 for r in recs if r["status"] == "keep"))
        mk = np.mean(kept)
        if g == "baseline":
            p_gap = p_test = float("nan")
        else:
            b_gap = np.array([base[t][2] for t in shared])
            b_dtest = np.array([base[t][1] for t in shared])
            dgap = b_gap - gap  # positive = judge closes gap
            dtt = dtest - b_dtest  # positive = judge improves test
            p_gap = (
                stats.wilcoxon(dgap, alternative="greater").pvalue
                if (dgap != 0).any()
                else float("nan")
            )
            p_test = (
                stats.wilcoxon(dtt, alternative="greater").pvalue
                if (dtt != 0).any()
                else float("nan")
            )
        print(
            f"{g:18s} {m_dval:+.4f} {m_dtest:+.4f} {m_gap:+.4f} {mk:5.1f} "
            f"{p_gap:11.4f} {p_test:13.4f}"
        )

    # per-task gap table: baseline vs effect_size vs statistical judges
    print("\nPer-task gap (Δval−Δtest):")
    print(f"{'task':34s} {'base':>8s} {'eff_k1':>8s} {'j_noise':>8s} {'j_hybrid':>9s}")
    for t in tasks:
        print(
            f"{t:34s} {base[t][2]:+.4f} {per_gate['effect_size_k1.0'][t][2]:+.4f} "
            f"{per_gate['judge_noise'][t][2]:+.4f} {per_gate['judge_hybrid'][t][2]:+.4f}"
        )


if __name__ == "__main__":
    main()
