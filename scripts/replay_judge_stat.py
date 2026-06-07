"""Statistical-strength LLM-as-judge gates, replayed over cached proposals.

Unlike the complexity judge (which scored the *code diff*), these judges score
the *evidence* for each proposed val improvement — the signal the effect-size
gate exploits — delivered through an LLM-as-judge framing.

Two gates:
  judge_noise  (Variant 1, pure reasoning): the judge sees the val improvement,
      val-set size, class structure, #experiments-so-far, and train-vs-val
      movement, and estimates the probability the gain is real (not val noise).
      It is NOT given the bootstrap sigma — it must reason the noise floor itself.
  judge_hybrid (Variant 2): same context PLUS the per-task bootstrap val sigma,
      so it can compare the gain directly to the measured noise.

A proposal is gated only when it improves on the current best val (Δval > 0),
matching baseline/effect_size semantics. The judge returns p_real in [0,1];
keep iff p_real >= 0.5. Decisions (and p_real) are cached per task+gate so the
replay is reproducible without re-calling the API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES, load_dataset  # noqa: E402

SIGMA_JSON = REPO / "results" / "_sigma" / "sigma.json"
MODEL = "claude-sonnet-4-5"
KEEP_THRESHOLD = 0.5

SYSTEM_NOISE = """You are a statistical gatekeeper for an autonomous ML research loop. It edits a \
train.py (XGBoost on one OpenML classification task), runs an experiment, and wants to keep only \
changes whose validation-error drop reflects a REAL, generalizable improvement — not noise from \
the finite validation set or from selecting the best of many tried experiments.

Validation error is a misclassification RATE measured on a fixed, finite validation set. A drop \
of D in that rate corresponds to only D * n_val samples changing their prediction. With a small \
val set, the standard error of a rate p is about sqrt(p*(1-p)/n_val); improvements smaller than \
~1-2 of those are easily noise. The more experiments have already been run, the more selection \
pressure inflates false positives (the best of many noisy trials looks good by luck).

Estimate the probability that THIS improvement is real. Respond with ONLY this JSON, no fences:
{"p_real": <float 0..1>, "reasoning": "<1-2 sentences>"}"""

SYSTEM_HYBRID = SYSTEM_NOISE + """

You are ALSO given sigma_val: the standard deviation of this task's validation error under \
bootstrap resampling of the validation set (the empirical noise floor). Compare the improvement \
to sigma_val directly when judging."""


def build_user(ctx: dict, with_sigma: bool) -> str:
    lines = [
        f"validation set size (n_val): {ctx['n_val']}",
        f"number of classes: {ctx['n_classes']}  (majority-class val error ~= {ctx['majority_err']:.3f})",
        f"experiments run so far: {ctx['n_experiments']}",
        f"current best val error: {ctx['best_val']:.4f}  (its train error: {ctx['best_train']:.4f})",
        f"proposed change val error: {ctx['val_err']:.4f}  (its train error: {ctx['train_err']:.4f})",
        f"val-error improvement (Δval): {ctx['dval']:.4f}  (= {ctx['dval'] * ctx['n_val']:.1f} samples on n_val)",
        f"change description: {ctx['desc']}",
    ]
    if with_sigma:
        lines.append(f"sigma_val (bootstrap val-error std): {ctx['sigma']:.4f}  (Δval / sigma_val = {ctx['dval'] / ctx['sigma']:.2f})")
    return "\n".join(lines)


def ask(client: anthropic.Anthropic, system: str, user: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=200, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "text", None)).strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    out = json.loads(text.strip())
    assert 0.0 <= out["p_real"] <= 1.0, out
    return out


def task_meta(task: str) -> dict:
    d = load_dataset(task)
    _, counts = np.unique(d.y_val, return_counts=True)
    return {"n_val": int(len(d.y_val)), "n_classes": int(d.n_classes),
            "majority_err": float(1 - counts.max() / len(d.y_val))}


def run_gate(task: str, gate: str, client: anthropic.Anthropic) -> None:
    with_sigma = gate == "judge_hybrid"
    system = SYSTEM_HYBRID if with_sigma else SYSTEM_NOISE
    meta = task_meta(task)
    sigma = json.loads(SIGMA_JSON.read_text())["per_task"][task]["bootstrap_val_std"]
    records = json.loads((REPO / "results" / task / "records.json").read_text())
    rows = [r for r in records if r["val_err"] == r["val_err"]]

    out_dir = REPO / "results" / task / gate
    out_dir.mkdir(parents=True, exist_ok=True)
    dec_path = out_dir / "decisions.json"
    decisions: dict[str, dict] = json.loads(dec_path.read_text()) if dec_path.exists() else {}

    best_val = rows[0]["val_err"]
    best_test = rows[0]["test_err"]
    best_train = rows[0]["train_err"]
    out = [{"i": rows[0]["i"], "desc": rows[0]["desc"], "train_err": best_train,
            "val_err": best_val, "test_err": best_test, "status": "keep",
            "best_val_after": best_val, "best_test_after": best_test}]
    n_eval = 1
    for r in rows[1:]:
        n_eval += 1
        dval = best_val - r["val_err"]
        if dval <= 0:
            status = "discard"
        else:
            key = str(r["i"])
            if key not in decisions:
                ctx = {**meta, "n_experiments": n_eval, "best_val": best_val,
                       "best_train": best_train, "val_err": r["val_err"],
                       "train_err": r["train_err"], "dval": dval, "desc": r["desc"],
                       "sigma": sigma}
                decisions[key] = ask(client, system, build_user(ctx, with_sigma))
                dec_path.write_text(json.dumps(decisions, indent=2))
            keep = decisions[key]["p_real"] >= KEEP_THRESHOLD
            if keep:
                best_val, best_test, best_train = r["val_err"], r["test_err"], r["train_err"]
                status = "keep"
            else:
                status = "discard"
        out.append({"i": r["i"], "desc": r["desc"], "train_err": r["train_err"],
                    "val_err": r["val_err"], "test_err": r["test_err"], "status": status,
                    "best_val_after": best_val, "best_test_after": best_test})
    (out_dir / "records.json").write_text(json.dumps(out, indent=2))
    kept = sum(1 for r in out if r["status"] == "keep")
    print(f"{task:36s} {gate:13s} kept={kept:2d}/{len(out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="*", default=["judge_noise", "judge_hybrid"])
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for task in (args.tasks or TASK_NAMES):
        for gate in args.gates:
            run_gate(task, gate, client)


if __name__ == "__main__":
    main()
