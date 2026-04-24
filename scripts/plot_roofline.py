"""Aggregate plots used in WRITEUP.md.

Outputs (in results/_summary/):
  - baseline_diagonal.png         per-task Δval vs Δtest scatter (§3)
  - baseline_roofline.png         baseline-only mean Δval / Δtest trajectory (§3)
  - roofline_gap_overlay.png      mean Δval − Δtest gap per gate (§4.1)
  - roofline_abs_test_overlay.png mean test_err at val-best per gate (§4.1)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES  # noqa: E402


def load_gate(task: str, gate: str) -> list[dict]:
    if gate == "baseline":
        p = REPO / "results" / task / "records.json"
    elif gate == "noise_replay":
        p = REPO / "results" / "_noise" / task / "records.json"
    else:
        p = REPO / "results" / task / gate / "records.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def best_curves(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (val_best_so_far, test_at_val_best) over experiment index."""
    rows = [r for r in records if r["val_err"] == r["val_err"]]
    val_c = np.zeros(len(rows))
    test_c = np.zeros(len(rows))
    best_val = float("inf")
    best_test = float("nan")
    for i, r in enumerate(rows):
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


def mean_delta_curves(gate: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean (Δval, Δtest) across tasks at each experiment step, + std err.
    Δ is defined relative to experiment 0 (baseline) so positive = improvement."""
    dvals: list[np.ndarray] = []
    dtests: list[np.ndarray] = []
    for t in TASK_NAMES:
        recs = load_gate(t, gate)
        if not recs:
            continue
        val_c, test_c = best_curves(recs)
        if len(val_c) < 2:
            continue
        m = min(len(val_c), n)
        dvals.append(val_c[0] - val_c[:m])
        dtests.append(test_c[0] - test_c[:m])

    # Pad to n with the last value (no more experiments for that task).
    def pad(arrs: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(arrs), n))
        for i, a in enumerate(arrs):
            out[i, : len(a)] = a
            if len(a) < n:
                out[i, len(a) :] = a[-1]
        return out

    A = pad(dvals)
    B = pad(dtests)
    return (
        A.mean(axis=0),
        A.std(axis=0, ddof=1) / np.sqrt(len(A)),
        B.mean(axis=0),
        B.std(axis=0, ddof=1) / np.sqrt(len(B)),
    )


GATE_ORDER = [
    "baseline",
    "effect_size_k1.0",
    "topk_confirm_k3",
    "thresholdout_b10",
    "rotating_val",
    "reflection",
    "noise_replay",
]
N_STEPS = 25


def plot_gap_overlay(out: Path) -> None:
    """Single axes. x = experiment index, y = Δval − Δtest (overfit gap),
    one line per gate. Lower = less overfit."""
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, g in enumerate(GATE_ORDER):
        mv, _, mt, _ = mean_delta_curves(g, N_STEPS)
        gap = mv - mt
        ls = "-" if g in ("baseline", "reflection", "noise_replay") else "--"
        lw = 2.6 if g in ("baseline", "effect_size_k1.0", "reflection") else 1.5
        ax.plot(
            np.arange(N_STEPS),
            gap,
            color=colors[i % len(colors)],
            lw=lw,
            linestyle=ls,
            label=f"{g}  (final={gap[-1]:+.4f})",
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3)
    ax.set_xlabel("experiment index")
    ax.set_ylabel("Δval − Δtest  (overfit gap, mean over 15 tasks)")
    ax.set_title(
        "Overfit gap per gate: Δval minus Δtest at the val-best experiment\n"
        "Zero = gate-selected wins generalize perfectly. Positive = val-gate cheating."
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def mean_abs_test(gate: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute test_err at val-best, across 15 tasks, at each exp index."""
    per_task: list[np.ndarray] = []
    for t in TASK_NAMES:
        recs = load_gate(t, gate)
        if not recs:
            continue
        _, test_c = best_curves(recs)
        if len(test_c) < 2:
            continue
        m = min(len(test_c), n)
        padded = np.concatenate([test_c[:m], np.full(n - m, test_c[-1] if len(test_c) else np.nan)])
        per_task.append(padded)
    A = np.stack(per_task)
    return A.mean(axis=0), A.std(axis=0, ddof=1) / np.sqrt(len(A))


def plot_abs_test_overlay(out: Path) -> None:
    """Single axes, absolute test_err at val-best, averaged over 15 tasks."""
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, g in enumerate(GATE_ORDER):
        m, se = mean_abs_test(g, N_STEPS)
        ls = "-" if g in ("baseline", "reflection", "noise_replay") else "--"
        lw = 2.6 if g in ("baseline", "effect_size_k1.0", "reflection") else 1.5
        ax.plot(
            np.arange(N_STEPS),
            m,
            color=colors[i % len(colors)],
            lw=lw,
            linestyle=ls,
            label=f"{g}  (final={m[-1]:.4f})",
        )
    ax.grid(alpha=0.3)
    ax.set_xlabel("experiment index")
    ax.set_ylabel("absolute test_err at val-best  (mean over 15 tasks)")
    ax.set_title(
        "Absolute test error per gate: mean test_err at the val-best experiment so far\n"
        "Lower = better. Flat or rising = gate picked wins that did not generalize."
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def plot_baseline_diagonal(out: Path) -> None:
    """Single-panel Δval vs Δtest scatter for baseline only, labeled by task."""
    dvs: list[float] = []
    dts: list[float] = []
    names: list[str] = []
    for t in TASK_NAMES:
        recs = load_gate(t, "baseline")
        if not recs:
            continue
        val_c, test_c = best_curves(recs)
        if len(val_c) < 2:
            continue
        dvs.append(val_c[0] - val_c[-1])
        dts.append(test_c[0] - test_c[-1])
        names.append(t)
    dv = np.array(dvs)
    dt = np.array(dts)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    lo, hi = -0.03, 0.10
    ax.plot([lo, hi], [lo, hi], color="gray", lw=1, linestyle="--", label="y = x (no overfit)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.fill_between([0, hi], [0, 0], [lo, lo], color="red", alpha=0.08)
    for x, y, n in zip(dv, dt, names, strict=True):
        overfit = x > 0 and y < x
        color = "C3" if (x > 0 and y < 0) else ("C1" if overfit else "C0")
        ax.scatter(x, y, s=70, color=color, edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(
            n,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
            alpha=0.85,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Δval (improvement the gate sees)")
    ax.set_ylabel("Δtest (ground-truth improvement)")
    ax.set_title(
        "Δval vs Δtest per task (baseline, 15 OpenML tasks)\n"
        "Red = val improved while test got worse.  "
        "Orange = val improved more than test (overfit).  Blue = honest."
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def plot_baseline_roofline(out: Path) -> None:
    mv, mv_se, mt, mt_se = mean_delta_curves("baseline", N_STEPS)
    xs = np.arange(N_STEPS)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.fill_between(xs, mv - mv_se, mv + mv_se, color="C0", alpha=0.2)
    ax.fill_between(xs, mt - mt_se, mt + mt_se, color="C1", alpha=0.2)
    ax.plot(xs, mv, color="C0", lw=2, label=f"Δval (what the gate sees), final {mv[-1]:+.4f}")
    ax.plot(xs, mt, color="C1", lw=2, label=f"Δtest (ground truth),      final {mt[-1]:+.4f}")
    ax.axhline(0, color="gray", lw=0.6)
    ax.grid(alpha=0.3)
    ax.set_xlabel("experiment index")
    ax.set_ylabel("Δ vs experiment 0 (positive = better)")
    ax.set_title(
        "Baseline roofline: mean across 15 tasks, shaded ± 1 SE. "
        f"Overfit gap at end: {mv[-1] - mt[-1]:+.4f}"
    )
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    out_dir = REPO / "results" / "_summary"
    plot_baseline_diagonal(out_dir / "baseline_diagonal.png")
    plot_baseline_roofline(out_dir / "baseline_roofline.png")
    plot_gap_overlay(out_dir / "roofline_gap_overlay.png")
    plot_abs_test_overlay(out_dir / "roofline_abs_test_overlay.png")


if __name__ == "__main__":
    main()
