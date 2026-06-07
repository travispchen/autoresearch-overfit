"""Can we replace the bootstrap val-noise sigma — the load-bearing ingredient of
the effect-size gate — without running a bootstrap?

The bootstrap sigma in sigma.json is the std of a *misclassification rate* under
val-slice resampling. That is, by construction, the binomial standard error of a
rate, so it has a closed form from just (n_val, p):

    binom_sigma     = sqrt(p*(1-p)/n_val)                       # frequentist SE
    jeffreys_sigma  = SD of Beta(k+0.5, n_val-k+0.5), k=round(p*n_val)   # Bayesian

This script also asks an LLM to estimate the same sigma from (n_val, n_classes,
majority_err, val_err), to test whether a judge can reason its way to the noise
floor instead of being handed it.

Each sigma source is then plugged into the SAME effect-size gate (k=1.0, the
published rule) so the only thing that varies is where sigma came from. We
compare the resulting gates to the bootstrap-sigma gate (effect_size_k1.0) on
the small-val tasks where the LLM-without-sigma judge (judge_noise) failed.

LLM estimates are cached in results/_sigma/sigma_llm.json so the gates reproduce
without re-calling the API (needs ANTHROPIC_API_KEY only to populate the cache).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import anthropic
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES, load_dataset  # noqa: E402

SIGMA_JSON = REPO / "results" / "_sigma" / "sigma.json"
LLM_SIGMA_JSON = REPO / "results" / "_sigma" / "sigma_llm.json"
MODEL = "claude-sonnet-4-5"
K = 1.0

SYSTEM_SIGMA = """You estimate the noise floor of a validation metric for an ML research loop.

A model is evaluated on a FIXED validation set of n_val examples. Validation error is the \
misclassification RATE on that set. If the validation set had instead been a different random \
draw of n_val examples from the same distribution, the measured error would wiggle. Estimate \
the standard deviation (sigma) of the validation error rate under that resampling.

Respond with ONLY this JSON, no fences:
{"sigma": <float>, "reasoning": "<1 sentence>"}"""


def binom_sigma(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def jeffreys_sigma(p: float, n: int) -> float:
    k = round(p * n)
    a, b = k + 0.5, n - k + 0.5
    return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))


def llm_sigma(
    client: anthropic.Anthropic, n_val: int, n_classes: int, majority_err: float, val_err: float
) -> dict:
    user = (
        f"n_val: {n_val}\n"
        f"number of classes: {n_classes}\n"
        f"majority-class baseline error: {majority_err:.4f}\n"
        f"current model validation error: {val_err:.4f}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM_SIGMA,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "text", None)).strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    out = json.loads(text.strip())
    assert out["sigma"] >= 0.0, out
    return out


def task_meta(task: str) -> dict:
    d = load_dataset(task)
    _, counts = np.unique(d.y_val, return_counts=True)
    return {"n_classes": int(d.n_classes), "majority_err": float(1 - counts.max() / len(d.y_val))}


def gate_effect_size(task: str, sigma: float) -> list[dict]:
    """k=1.0 effect-size gate over cached proposals, identical to the published
    rule except sigma is supplied by the caller."""
    records = json.loads((REPO / "results" / task / "records.json").read_text())
    rows = [r for r in records if r["val_err"] == r["val_err"]]
    best_val, best_test = rows[0]["val_err"], rows[0]["test_err"]
    out = [
        {
            "i": rows[0]["i"],
            "desc": rows[0]["desc"],
            "train_err": rows[0]["train_err"],
            "val_err": best_val,
            "test_err": best_test,
            "status": "keep",
            "best_val_after": best_val,
            "best_test_after": best_test,
        }
    ]
    for r in rows[1:]:
        if best_val - r["val_err"] > K * sigma:
            best_val, best_test = r["val_err"], r["test_err"]
            status = "keep"
        else:
            status = "discard"
        out.append(
            {
                "i": r["i"],
                "desc": r["desc"],
                "train_err": r["train_err"],
                "val_err": r["val_err"],
                "test_err": r["test_err"],
                "status": status,
                "best_val_after": best_val,
                "best_test_after": best_test,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--no-llm", action="store_true", help="skip LLM sigma (use cache only)")
    args = ap.parse_args()

    per_task = json.loads(SIGMA_JSON.read_text())["per_task"]
    llm_cache = json.loads(LLM_SIGMA_JSON.read_text()) if LLM_SIGMA_JSON.exists() else {}
    client = None if args.no_llm else anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(
        f"{'task':34s} {'n_val':>5s} {'boot':>8s} {'binom':>8s} {'jeff':>8s} {'llm':>8s} "
        f"{'b/boot':>7s} {'j/boot':>7s} {'l/boot':>7s}"
    )
    rows = []
    for task in args.tasks or TASK_NAMES:
        t = per_task[task]
        n, p, boot = t["n_val"], t["baseline_val_err"], t["bootstrap_val_std"]
        sig_b = binom_sigma(p, n)
        sig_j = jeffreys_sigma(p, n)
        if task not in llm_cache:
            assert client is not None, f"{task} not in LLM sigma cache and --no-llm set"
            meta = task_meta(task)
            llm_cache[task] = llm_sigma(client, n, meta["n_classes"], meta["majority_err"], p)
            LLM_SIGMA_JSON.write_text(json.dumps(llm_cache, indent=2))
        sig_l = llm_cache[task]["sigma"]
        rb = sig_b / boot if boot > 0 else float("nan")
        rj = sig_j / boot if boot > 0 else float("nan")
        rl = sig_l / boot if boot > 0 else float("nan")
        print(
            f"{task:34s} {n:5d} {boot:8.5f} {sig_b:8.5f} {sig_j:8.5f} {sig_l:8.5f} "
            f"{rb:7.3f} {rj:7.3f} {rl:7.3f}"
        )
        rows.append((task, n))

        for gate, sigma in [
            ("sigma_binom", sig_b),
            ("sigma_jeffreys", sig_j),
            ("sigma_llm", sig_l),
        ]:
            recs = gate_effect_size(task, sigma)
            out_dir = REPO / "results" / task / gate
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "records.json").write_text(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
