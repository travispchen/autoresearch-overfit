"""LLM-as-judge complexity-regularization gate, replayed over cached proposals.

Tests the hypothesis: instead of a uniform effect-size margin (k*sigma for
every proposal), let an LLM score *each proposed code change* on two axes —
complexity and overfitting_risk — and require a val improvement proportional
to that score. Cheap simplifications pass with ~no margin; risky/complex
changes must clear a larger bar.

Per proposal i, the "change" scored is the incremental unified diff from the
train.py the agent was editing (the baseline-gate kept trajectory, which is
fixed) to proposal i's train.py. That makes the score a property of the
proposal alone, so it is cached per (task, i) and reused across thresholds.

Gate rule (complexity-adaptive effect-size gate):
    penalty_i  = (complexity_i + overfitting_risk_i) / 2          in [0, 1]
    margin_i   = penalty_i * lambda_scale * sigma_task
    accept iff best_val - val_err > margin_i

With lambda_scale=2 and mean penalty ~0.5 the mean margin is ~1*sigma, so this
is the same average strictness as effect_size_k1.0 but the margin is allocated
by judged complexity rather than uniformly.

Judge scores are cached in results/<task>/judge_scores.json so the gate is
reproducible without re-calling the API. Needs ANTHROPIC_API_KEY only to
populate that cache the first time.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

import anthropic

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES  # noqa: E402

SIGMA_JSON = REPO / "results" / "_sigma" / "sigma.json"
MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a complexity judge for an autonomous ML research loop that edits a \
train.py (an XGBoost classifier on one OpenML tabular task), runs an experiment, and keeps \
changes that lower validation error (val_err).

Assess each diff on two axes (0.0 to 1.0):

COMPLEXITY — how much complexity does the change add?
  0.0 = simplification (deleting code, fewer knobs)
  0.3 = minor (one constant changed)
  0.5 = moderate (new mechanism, few new params)
  0.7 = significant (multiple new components)
  1.0 = complete rewrite

OVERFITTING_RISK — how likely to overfit the val metric through selection?
  0.0 = reduces risk (regularization, fewer degrees of freedom)
  0.3 = low (well-known technique, no new tunable params)
  0.5 = moderate (a few new hyperparameters)
  0.7 = high (many new degrees of freedom)
  1.0 = extreme (appears to game the metric)

Respond with ONLY this JSON, no markdown fences:
{"complexity": <float>, "overfitting_risk": <float>, "reasoning": "<1-2 sentences>"}"""


def unified_diff(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="train.py",
            tofile="train.py",
        )
    )


def judge_diff(client: anthropic.Anthropic, diff: str) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"```diff\n{diff}\n```"}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "text", None)).strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    out = json.loads(text.strip())
    assert 0.0 <= out["complexity"] <= 1.0 and 0.0 <= out["overfitting_risk"] <= 1.0
    return out


def score_task(task: str, client: anthropic.Anthropic | None) -> dict[int, dict]:
    """Return {i: {complexity, overfitting_risk, penalty, reasoning}} for every
    non-baseline proposal with a real val_err. Cached on disk."""
    task_dir = REPO / "results" / task
    props = {
        p["i"]: p["new_train_py"]
        for p in (
            json.loads(line) for line in (task_dir / "proposals.jsonl").read_text().splitlines()
        )
    }
    records = json.loads((task_dir / "records.json").read_text())

    cache_path = task_dir / "judge_scores.json"
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    kept_src = props[0]  # baseline-gate kept trajectory (fixed)
    scores: dict[int, dict] = {}
    for r in records[1:]:
        i = r["i"]
        if r["val_err"] != r["val_err"]:  # nan — errored experiment, ungated
            continue
        diff = unified_diff(kept_src, props[i])
        if str(i) not in cache:
            assert client is not None, f"{task} i={i} not cached and no API client"
            assert diff.strip(), f"{task} i={i}: empty diff"
            cache[str(i)] = judge_diff(client, diff)
            cache_path.write_text(json.dumps(cache, indent=2))
        s = cache[str(i)]
        scores[i] = {**s, "penalty": (s["complexity"] + s["overfitting_risk"]) / 2}
        if r["status"] == "keep":
            kept_src = props[i]
    return scores


def gate_judge(task: str, scores: dict[int, dict], sigma: float, lam: float) -> list[dict]:
    records = json.loads((REPO / "results" / task / "records.json").read_text())
    rows = [r for r in records if r["val_err"] == r["val_err"]]
    best_val = rows[0]["val_err"]
    best_test = rows[0]["test_err"]
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
        margin = scores[r["i"]]["penalty"] * lam * sigma
        if best_val - r["val_err"] > margin:
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
    ap.add_argument("--lambdas", type=float, nargs="*", default=[1.0, 2.0, 3.0])
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--score-only", action="store_true", help="populate score cache, no gating")
    args = ap.parse_args()

    sigma_data = json.loads(SIGMA_JSON.read_text())["per_task"]
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for task in args.tasks or TASK_NAMES:
        scores = score_task(task, client)
        n = len(scores)
        mean_pen = sum(s["penalty"] for s in scores.values()) / n if n else float("nan")
        print(f"{task:36s} scored {n:3d} proposals  mean_penalty={mean_pen:.3f}")
        if args.score_only:
            continue
        sigma = sigma_data[task]["bootstrap_val_std"]
        for lam in args.lambdas:
            recs = gate_judge(task, scores, sigma, lam)
            gate = f"judge_k{lam}"
            out_dir = REPO / "results" / task / gate
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "records.json").write_text(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
