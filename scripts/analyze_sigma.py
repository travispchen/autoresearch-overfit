"""Compare bootstrap-free sigma sources against the bootstrap-sigma gate.

All gates here are the SAME k=1.0 effect-size rule; only the sigma source differs:
  effect_size_k1.0  -> bootstrap sigma (the published reference)
  sigma_binom       -> closed-form binomial SE  sqrt(p(1-p)/n_val)
  sigma_jeffreys    -> Beta(Jeffreys) posterior SD
  sigma_llm         -> LLM-estimated sigma
For contrast we also show judge_noise (LLM keep/reject WITHOUT being given sigma).

Reported: mean gap/Δtest/kept overall and split by val-set size, plus paired
Wilcoxon of each gate's per-task gap vs the bootstrap gate (two-sided: are they
distinguishable?).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from stats_significance import per_task_stats  # noqa: E402

REF = "effect_size_k1.0"
GATES = ["baseline", REF, "sigma_binom", "sigma_jeffreys", "sigma_llm", "judge_noise"]
SMALL_VAL_MAX = 150


def kept_count(task: str, gate: str) -> int:
    p = (
        (REPO / "results" / task / "records.json")
        if gate == "baseline"
        else (REPO / "results" / task / gate / "records.json")
    )
    return sum(1 for r in json.loads(p.read_text()) if r["status"] == "keep")


def main() -> None:
    per_task = json.loads((REPO / "results" / "_sigma" / "sigma.json").read_text())["per_task"]
    stats_by_gate = {g: per_task_stats(g) for g in GATES}
    tasks = sorted(stats_by_gate[REF].keys())
    small = [t for t in tasks if per_task[t]["n_val"] <= SMALL_VAL_MAX]
    large = [t for t in tasks if per_task[t]["n_val"] > SMALL_VAL_MAX]

    def mean_gap(gate: str, ts: list[str]) -> float:
        return float(np.mean([stats_by_gate[gate][t][2] for t in ts]))

    print(
        f"{'gate':18s} {'μgap(all)':>10s} {'μΔtest':>8s} {'kept':>5s} "
        f"{'μgap small':>11s} {'μgap large':>11s} {'p vs boot':>10s}"
    )
    print("-" * 80)
    ref = stats_by_gate[REF]
    for g in GATES:
        gs = stats_by_gate[g]
        gap_all = mean_gap(g, tasks)
        dtest = float(np.mean([gs[t][1] for t in tasks]))
        kept = float(np.mean([kept_count(t, g) for t in tasks]))
        gp_s = mean_gap(g, small)
        gp_l = mean_gap(g, large)
        if g in (REF, "baseline"):
            p = float("nan")
        else:
            d = np.array([gs[t][2] - ref[t][2] for t in tasks])
            p = stats.wilcoxon(d).pvalue if (d != 0).any() else float("nan")
        print(
            f"{g:18s} {gap_all:+10.4f} {dtest:+8.4f} {kept:5.1f} "
            f"{gp_s:+11.4f} {gp_l:+11.4f} {p:10.4f}"
        )

    print(f"\nsmall-val tasks (n_val<={SMALL_VAL_MAX}): {', '.join(small)}")
    print("\nPer-task gap on small-val tasks:")
    print(
        f"{'task':34s} {'n_val':>5s} {'base':>8s} {'boot':>8s} {'binom':>8s} {'llm':>8s} {'j_noise':>8s}"
    )
    for t in small:
        n = per_task[t]["n_val"]
        print(
            f"{t:34s} {n:5d} {stats_by_gate['baseline'][t][2]:+.4f} {ref[t][2]:+.4f} "
            f"{stats_by_gate['sigma_binom'][t][2]:+.4f} {stats_by_gate['sigma_llm'][t][2]:+.4f} "
            f"{stats_by_gate['judge_noise'][t][2]:+.4f}"
        )


if __name__ == "__main__":
    main()
