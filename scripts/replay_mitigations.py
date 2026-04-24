"""Replay each task's baseline proposal stream through different gates.

For gate-side mitigations we don't re-run the LLM. We walk through each
task's `records.json` in order and apply a different accept/reject rule.
The stored (val_err, test_err) per proposal is the SAME values the baseline
run observed, so each gate sees the same candidate set — only the keep
decision differs. True apples to apples.

Writes `results/<task>/<gate>/records.json` per task per gate.

Gates implemented:
  - effect_size_k{k}        : accept if best_val - val_err > k * sigma_val
  - thresholdout_b{budget}  : noisy reveal with a reveal-budget (Dwork 2015)
  - rotating_val            : partition val into K disjoint folds, evaluate
                              experiment i on fold (i mod K), track per-fold best
  - multi_seed_avg_m{m}     : accept if mean val over M sampled seeds < best
  - train_val_agree         : accept if val improves AND train also improves
                              (direction), with tolerance tau
  - topk_confirm_k{k}       : final re-eval of top-K kept configs on a bootstrap
                              holdout drawn from train+val pool

For gates that need signals the baseline run didn't produce (multi-seed
averages, train+val bootstrap), we use the sigma estimate + baseline train
errors to simulate deterministically — these are clearly labeled as
approximations in the code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES, decode_val_per_sample, silent_log_path  # noqa: E402

SIGMA_JSON = REPO / "results" / "_sigma" / "sigma.json"


@dataclass
class Step:
    i: int
    desc: str
    train_err: float
    val_err: float
    test_err: float
    status: str
    val_per_sample: np.ndarray | None  # per-sample val error (squared for reg, 0/1 for cls)


def load_silent_per_sample(task: str) -> dict[str, np.ndarray]:
    """Build train_py_hash -> val_per_sample from the silent log (latest entry per hash)."""
    p = silent_log_path(task)
    assert p.exists(), f"no silent log for {task}"
    out: dict[str, np.ndarray] = {}
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            if "val_per_sample_b64" in r and "train_py_hash" in r:
                out[r["train_py_hash"]] = decode_val_per_sample(r["val_per_sample_b64"], r["n_val"])
    return out


def load_task_records(task: str) -> list[Step]:
    p = REPO / "results" / task / "records.json"
    records = json.loads(p.read_text())
    per_sample = load_silent_per_sample(task)
    out: list[Step] = []
    for r in records:
        # Skip errored experiments — they have nan val/test and can't be gated.
        if r["val_err"] != r["val_err"]:
            continue
        h = r.get("train_py_hash")
        ps = per_sample.get(h) if h else None
        out.append(
            Step(
                i=r["i"],
                desc=r["desc"],
                train_err=r["train_err"],
                val_err=r["val_err"],
                test_err=r["test_err"],
                status=r["status"],
                val_per_sample=ps,
            )
        )
    return out


def _emit(step: Step, status: str, best_val: float, best_test: float) -> dict:
    return {
        "i": step.i,
        "desc": step.desc,
        "train_err": step.train_err,
        "val_err": step.val_err,
        "test_err": step.test_err,
        "status": status,
        "best_val_after": best_val,
        "best_test_after": best_test,
    }


def _keep_first(steps: list[Step]) -> tuple[float, float, list[dict]]:
    best_val = steps[0].val_err
    best_test = steps[0].test_err
    out = [_emit(steps[0], "keep", best_val, best_test)]
    return best_val, best_test, out


# ----- gates -----


def gate_baseline(steps: list[Step]) -> list[dict]:
    best_val, best_test, out = _keep_first(steps)
    for s in steps[1:]:
        if s.val_err < best_val:
            best_val, best_test = s.val_err, s.test_err
            status = "keep"
        else:
            status = "discard"
        out.append(_emit(s, status, best_val, best_test))
    return out


def gate_effect_size(steps: list[Step], sigma: float, k: float) -> list[dict]:
    best_val, best_test, out = _keep_first(steps)
    margin = k * sigma
    for s in steps[1:]:
        if best_val - s.val_err > margin:
            best_val, best_test = s.val_err, s.test_err
            status = "keep"
        else:
            status = "discard"
        out.append(_emit(s, status, best_val, best_test))
    return out


def gate_thresholdout(
    steps: list[Step], sigma: float, budget: int, tau: float, rng: np.random.Generator
) -> list[dict]:
    """Dwork-style thresholdout: noisy reveal with budget.

    Gate returns 'noisy val close to best' unless the candidate is far enough
    away (|diff| > tau + Laplace noise). Each far reveal spends budget.
    """
    best_val, best_test, out = _keep_first(steps)
    remaining = budget
    noisy_val_of_best = best_val + rng.laplace(0.0, sigma)
    for s in steps[1:]:
        diff = best_val - s.val_err
        gamma = rng.laplace(0.0, sigma)
        if remaining > 0 and (diff + gamma) > tau:
            # Reveal the true val_err of the candidate, relative to best.
            remaining -= 1
            if s.val_err < best_val:
                best_val, best_test = s.val_err, s.test_err
                noisy_val_of_best = best_val + rng.laplace(0.0, sigma)
                status = "keep"
            else:
                status = "discard"
        else:
            # Use the *noisy stand-in* — cannot distinguish reliably, discard.
            status = "discard"
            _ = noisy_val_of_best  # keep the variable alive for readability
        out.append(_emit(s, status, best_val, best_test))
    return out


def gate_train_val_agree(steps: list[Step], tau: float) -> list[dict]:
    """Accept only if val improves AND train_err doesn't get *worse* by more
    than tau. Requires both to move in the same direction (down)."""
    best_val, best_test, out = _keep_first(steps)
    best_train = steps[0].train_err
    for s in steps[1:]:
        if s.val_err < best_val and s.train_err <= best_train + tau:
            best_val, best_test, best_train = s.val_err, s.test_err, s.train_err
            status = "keep"
        else:
            status = "discard"
        out.append(_emit(s, status, best_val, best_test))
    return out


def gate_multi_seed_avg(
    steps: list[Step], sigma: float, M: int, rng: np.random.Generator
) -> list[dict]:
    """Approximate: draw M noisy observations centered on the true val_err
    (with sigma from the baseline's seed distribution), average, and gate.
    Realistic because averaging M iid noisy views cuts the noise by sqrt(M).
    """
    best_val, best_test, out = _keep_first(steps)
    # Establish a fresh noisy 'best' estimate too — apples to apples.
    best_est = best_val + rng.normal(0.0, sigma / np.sqrt(M))
    for s in steps[1:]:
        est = s.val_err + rng.normal(0.0, sigma / np.sqrt(M))
        if est < best_est:
            best_val, best_test, best_est = s.val_err, s.test_err, est
            status = "keep"
        else:
            status = "discard"
        out.append(_emit(s, status, best_val, best_test))
    return out


def gate_rotating_val(steps: list[Step], rng: np.random.Generator) -> list[dict]:
    """True rotating val: partition the val slice into K disjoint folds and
    evaluate each experiment on a different fold (cycling through).

    Gate decision uses fold-val: experiment i is evaluated on fold (i mod K)
    with per-fold running best. Because folds are disjoint and each fold is
    only revisited every K experiments, the gate cannot memorize any single
    sample the way the fixed-val baseline does.

    For reporting, `best_val_after` tracks the lowest FULL-val across all
    kept-so-far experiments and `best_test_after` is the test_err of that
    full-val argmin experiment. This makes rotating_val comparable to the
    other gates on the same Δval / Δtest / gap axes. The intended final
    selection rule is "among all gate-accepted experiments, pick the one
    with the lowest full-val".

    K is chosen per task to keep each fold around 50 samples, capped at 5
    to avoid excessive cycle length given ~25 experiments.
    """
    assert steps[0].val_per_sample is not None, "baseline step missing per-sample val"
    n_val = steps[0].val_per_sample.shape[0]
    K = min(5, max(2, n_val // 50))
    perm = rng.permutation(n_val)
    folds = np.array_split(perm, K)

    base_ps = steps[0].val_per_sample
    best_per_fold: list[float] = [float("inf")] * K
    best_per_fold[0] = float(base_ps[folds[0]].mean())
    best_val = steps[0].val_err
    best_test = steps[0].test_err
    out = [_emit(steps[0], "keep", best_val, best_test)]

    for idx, s in enumerate(steps[1:], start=1):
        k = idx % K
        assert s.val_per_sample is not None, f"step {idx} missing per-sample val"
        assert s.val_per_sample.shape[0] == n_val
        fold_err = float(s.val_per_sample[folds[k]].mean())
        if fold_err < best_per_fold[k]:
            best_per_fold[k] = fold_err
            status = "keep"
            if s.val_err < best_val:
                best_val = s.val_err
                best_test = s.test_err
        else:
            status = "discard"
        out.append(_emit(s, status, best_val, best_test))
    return out


def gate_topk_confirm(
    steps: list[Step], sigma: float, K: int, rng: np.random.Generator
) -> list[dict]:
    """Greedy baseline gate, then re-evaluate top-K kept configs on a
    bootstrap holdout (simulated by adding fresh noise), pick the best.
    """
    out = gate_baseline(steps)
    kept_steps = [(s, o) for s, o in zip(steps, out, strict=True) if o["status"] == "keep"]
    if len(kept_steps) <= 1:
        return out
    # Select top-K by observed val_err (lowest val wins), then re-eval those K on
    # a fresh slice and pick the best reeval. With K=len(kept_steps) this
    # degenerates to re-evaluating all kept configs.
    kept_steps.sort(key=lambda z: z[0].val_err)
    topk = kept_steps[:K]
    scored = [(s, o, s.val_err + rng.normal(0.0, sigma)) for s, o in topk]
    scored.sort(key=lambda z: z[2])
    winner_step, _, _ = scored[0]
    # Rewrite best_val_after/best_test_after to reflect the final survivor.
    survivor_val = winner_step.val_err
    survivor_test = winner_step.test_err
    # Walk the trajectory; until we reach survivor's i, use the greedy best;
    # from survivor's i onward, freeze at survivor.
    final: list[dict] = []
    greedy_best_val = steps[0].val_err
    greedy_best_test = steps[0].test_err
    survivor_reached = False
    for s, o in zip(steps, out, strict=True):
        if o["status"] == "keep":
            greedy_best_val = s.val_err
            greedy_best_test = s.test_err
        if s.i == winner_step.i:
            survivor_reached = True
        if survivor_reached:
            new = dict(o)
            new["best_val_after"] = survivor_val
            new["best_test_after"] = survivor_test
            final.append(new)
        else:
            new = dict(o)
            new["best_val_after"] = greedy_best_val
            new["best_test_after"] = greedy_best_test
            final.append(new)
    return final


# ----- driver -----


GATES = {
    "baseline_gate": lambda steps, sig, rng: gate_baseline(steps),
    "effect_size_k0.5": lambda steps, sig, rng: gate_effect_size(steps, sig, 0.5),
    "effect_size_k1.0": lambda steps, sig, rng: gate_effect_size(steps, sig, 1.0),
    "effect_size_k2.0": lambda steps, sig, rng: gate_effect_size(steps, sig, 2.0),
    "thresholdout_b3": lambda steps, sig, rng: gate_thresholdout(steps, sig, 3, 0.5 * sig, rng),
    "thresholdout_b5": lambda steps, sig, rng: gate_thresholdout(steps, sig, 5, 0.5 * sig, rng),
    "thresholdout_b10": lambda steps, sig, rng: gate_thresholdout(steps, sig, 10, 0.5 * sig, rng),
    "train_val_agree_tau0.005": lambda steps, sig, rng: gate_train_val_agree(steps, 0.005),
    "multi_seed_avg_m3": lambda steps, sig, rng: gate_multi_seed_avg(steps, sig, 3, rng),
    "multi_seed_avg_m5": lambda steps, sig, rng: gate_multi_seed_avg(steps, sig, 5, rng),
    # rotating_val handled specially — it needs task_type, not sigma.
    "topk_confirm_k3": lambda steps, sig, rng: gate_topk_confirm(steps, sig, 3, rng),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="*", default=None, help="subset of gates")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert SIGMA_JSON.exists(), (
        f"{SIGMA_JSON} missing — run `uv run python scripts/estimate_sigma.py` first"
    )
    sigma_data = json.loads(SIGMA_JSON.read_text())
    sigma_per_task = {t: sigma_data["per_task"][t]["bootstrap_val_std"] for t in TASK_NAMES}

    gates = args.gates or list(GATES.keys()) + ["rotating_val"]
    for g in gates:
        assert g in GATES or g == "rotating_val", f"unknown gate: {g}"

    for task in TASK_NAMES:
        rec_path = REPO / "results" / task / "records.json"
        if not rec_path.exists():
            print(f"skip {task}: no records.json")
            continue
        steps = load_task_records(task)
        if not steps:
            continue
        sig = sigma_per_task[task]
        for g in gates:
            # Deterministic per-(task, gate) seed. Python's built-in hash() is
            # randomized per interpreter, so we use a stable sha256 digest.
            digest = hashlib.sha256(f"{task}:{g}".encode()).digest()
            seed = (args.seed + int.from_bytes(digest[:4], "big")) % (2**32)
            rng = np.random.default_rng(seed)
            if g == "rotating_val":
                out_recs = gate_rotating_val(steps, rng)
            else:
                out_recs = GATES[g](steps, sig, rng)
            out_dir = REPO / "results" / task / g
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "records.json").write_text(json.dumps(out_recs, indent=2))
        print(f"replayed {task}: {len(steps)} steps × {len(gates)} gates")


if __name__ == "__main__":
    main()
