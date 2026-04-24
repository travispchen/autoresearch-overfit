"""Proposer-side 'reflect on overfit' mitigation — a fresh per-task LLM loop.

Same contract as the baseline loop (read train.py + program.md + history,
propose ONE mutation, val-gate, keep/discard) but each prompt includes
explicit diagnostic signals:
  - train_err alongside val_err
  - Δtrain vs Δval for recent kept experiments
  - Running σ_val estimate (from results/_sigma/sigma.json)
  - Recent-keep pattern

And a stronger instruction: if the last win looks like noise (Δval ≤
0.5·σ_val, train flat), name that explicitly and propose a qualitatively
different kind of change. Require `[real]` or `[noise-avoid]` prefix in
the description so we can grep for reflection signal.

This is the proposer-side mitigation; it is NOT apples-to-apples with the
gate-side replays — those share the same proposal stream, this one draws
fresh proposals. Report separately.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare import TASK_NAMES, silent_log_path  # noqa: E402
from run_autoresearch import BASELINE_SRC, PROGRAM_MD  # noqa: E402

MODEL = "claude-sonnet-4-5"
MAX_TOKENS_PROPOSAL = 8192
SUBPROCESS_TIMEOUT = 300

_DESC_RE = re.compile(r"<description>\s*(.*?)\s*</description>", re.DOTALL)
_TRAIN_RE = re.compile(r"<train_py>\s*(.*?)\s*</train_py>", re.DOTALL)
_TRAIN_OPEN_RE = re.compile(r"<train_py>\s*(.*)", re.DOTALL)
_REFL_RE = re.compile(r"<reflection>\s*(.*?)\s*</reflection>", re.DOTALL)
_VAL_ERR_RE = re.compile(r"^val_err:\s*([0-9.]+)\s*$", re.MULTILINE)


def content_hash(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


SYSTEM_PROMPT_TEMPLATE = """You are an ML researcher running an autoresearch loop on ONE OpenML \
tabular classification task (name = "{task}"). Val_err is the mean misclassification rate on \
the task's val slice; lower is better. The starting `train.py` is an untuned XGBoost baseline.

Unlike a naive autoresearch loop, you receive explicit diagnostic signals each turn. Use them \
to detect noise-mining: a 'win' whose Δval is within the seed-to-seed noise floor, or whose \
Δtrain is ~0 (model behavior unchanged; the val delta is probably RNG).

{program_md}

### Reflection rules (proposer-side overfit mitigation)

Each turn, before proposing, examine the recent-kept pattern and the running σ estimate.

- If the last kept experiment has Δval ≤ 0.5 · σ_val AND Δtrain ≈ 0, it is almost certainly \
noise-mining. Name it in your reflection. Propose a QUALITATIVELY DIFFERENT kind of change \
next (different hyperparameter family, feature engineering, booster switch, ensembling) — \
NOT another small perturbation of the same config.

- If several recent kept experiments all moved val by < σ and left train unchanged, the \
trajectory is drifting. Propose a mutation that either increases model capacity, changes the \
booster, or meaningfully changes the feature representation, not another seed/colsample tweak.

- Prefix your description with `[real]` if you believe this is a real improvement, or \
`[noise-avoid]` if the change is primarily motivated by avoiding more noise-mining.

Output format (REQUIRED):

<reflection>
2–4 sentences: your honest read of recent trajectory. Does the last keep look like noise?
What kind of change will you propose and why?
</reflection>
<description>Short one-line description (≤ 20 words, no newlines). Prefix with [real] or \
[noise-avoid]. If the change uses eval_set with early_stopping_rounds, also add [es-on-val].</description>
<train_py>
# full new contents of train.py here, verbatim, ready to run via `TASK={task} uv run python train.py`
</train_py>

Do not include any other text outside those three tags. The train_py contents will be written \
verbatim to train.py.
"""


def build_user_message(
    task: str,
    current_src: str,
    best_val_err: float | None,
    history: list[dict[str, Any]],
    sigma_val: float,
) -> str:
    keeps = [h for h in history if h["status"] == "keep"]
    recent_keeps = keeps[-5:]
    recent_lines = []
    for rk in recent_keeps:
        dv = rk.get("delta_val", 0.0)
        dt = rk.get("delta_train", 0.0)
        recent_lines.append(
            f"  [{rk['i']:03d}] Δval={dv:+.4f} Δtrain={dt:+.4f}  train={rk['train_err']:.4f} "
            f"val={rk['val_err']:.4f}  {rk['desc']}"
        )
    keeps_block = "\n".join(recent_lines) if recent_lines else "  (none yet)"

    hist_lines = []
    for h in history[-10:]:
        v = f"{h['val_err']:.4f}" if h["val_err"] == h["val_err"] else "  nan"
        tr = f"{h['train_err']:.4f}" if h["train_err"] == h["train_err"] else "  nan"
        hist_lines.append(f"  [{h['i']:03d}] {h['status']:7s} val={v} train={tr}  {h['desc']}")
    hist = "\n".join(hist_lines) if hist_lines else "  (none yet)"
    best = f"{best_val_err:.4f}" if best_val_err is not None else "n/a"

    return f"""Task: {task}

Signals:
- running σ_val (seed-to-seed noise floor from 30 baseline retrains): {sigma_val:.5f}
- 0.5 * σ_val (below this, a win is likely noise): {0.5 * sigma_val:.5f}
- current best val_err: {best}

Recent KEPT experiments (most recent last):
{keeps_block}

Last 10 experiments (all statuses):
{hist}

Current train.py:
```python
{current_src}
```

Reflect, then propose ONE concrete mutation (qualitatively different if the last keep looks like noise).
"""


def parse_proposal(text: str) -> tuple[str, str, str]:
    dm = _DESC_RE.search(text)
    tm = _TRAIN_RE.search(text)
    rm = _REFL_RE.search(text)
    assert dm is not None, f"Missing <description> tag:\n{text[:500]}"
    desc = dm.group(1).strip().replace("\t", " ").replace("\n", " ")[:240]
    reflection = rm.group(1).strip() if rm else ""
    if tm is not None:
        new_train_py = tm.group(1)
    else:
        om = _TRAIN_OPEN_RE.search(text)
        assert om is not None, f"Missing <train_py> tag:\n{text[:500]}"
        new_train_py = om.group(1)
    new_train_py = re.sub(r"^```[a-zA-Z]*\n", "", new_train_py)
    new_train_py = re.sub(r"\n?```\s*$", "", new_train_py)
    return desc, reflection, new_train_py


def validate_train_py(src: str) -> tuple[bool, str]:
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {"X_test", "y_test"}:
            return False, f"forbidden name: {node.id}"
        if isinstance(node, ast.ImportFrom) and node.module == "prepare":
            for alias in node.names:
                if alias.name in {"_load_with_test", "silent_log_path"}:
                    return False, f"forbidden import from prepare: {alias.name}"
    if "evaluate_one" not in src:
        return False, "train.py must call evaluate_one"
    if 'TASK = os.environ["TASK"]' not in src and "TASK = os.environ['TASK']" not in src:
        return False, 'TASK = os.environ["TASK"] must be preserved'
    return True, ""


def run_train(task: str, task_dir: Path) -> float:
    env = os.environ.copy()
    env["TASK"] = task
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env.get("PYTHONPATH", "")).rstrip(os.pathsep)
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "--project", str(REPO), "python", str(task_dir / "train.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exit={proc.returncode}\n{proc.stderr[-1500:]}")
    m = _VAL_ERR_RE.search(proc.stdout)
    if not m:
        raise RuntimeError(f"could not parse val_err:\n{proc.stdout[-800:]}")
    return float(m.group(1))


def _tail(path: Path) -> dict[str, Any]:
    with path.open() as f:
        last = None
        for line in f:
            if line.strip():
                last = line
    assert last is not None
    return json.loads(last)


def eval_current(task: str, task_dir: Path, log_path: Path, cache: dict) -> dict:
    src = (task_dir / "train.py").read_text()
    h = content_hash(src)
    if h in cache:
        return cache[h]
    val_err = run_train(task, task_dir)
    tail = _tail(log_path)
    assert abs(tail["val_err"] - val_err) < 1e-4
    cache[h] = {
        "train_py_hash": h,
        "train_err": tail["train_err"],
        "val_err": tail["val_err"],
        "test_err": tail["test_err"],
    }
    return cache[h]


def run_one_task(task: str, n: int, sigma_val: float, seed: int = 0) -> tuple[str, int]:
    random.seed(seed)
    task_dir = REPO / "results" / task / "reflection"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "train.py").write_text(BASELINE_SRC)
    records_path = task_dir / "records.json"
    proposals_path = task_dir / "proposals.jsonl"
    cache_path = task_dir / "eval_cache.json"
    log_path = silent_log_path(task)

    cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ev0 = eval_current(task, task_dir, log_path, cache)
    cache_path.write_text(json.dumps(cache, indent=2))
    best_val = ev0["val_err"]
    best_train = ev0["train_err"]
    records: list[dict] = [
        {
            "i": 0,
            "desc": "baseline",
            "reflection": "",
            "train_err": ev0["train_err"],
            "val_err": ev0["val_err"],
            "test_err": ev0["test_err"],
            "delta_train": 0.0,
            "delta_val": 0.0,
            "status": "keep",
            "train_py_hash": ev0["train_py_hash"],
        }
    ]
    records_path.write_text(json.dumps(records, indent=2))
    with proposals_path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "i": 0,
                    "desc": "baseline",
                    "reflection": "",
                    "new_train_py": (task_dir / "train.py").read_text(),
                    "raw_llm": "",
                }
            )
            + "\n"
        )

    client = anthropic.Anthropic()
    system = SYSTEM_PROMPT_TEMPLATE.format(task=task, program_md=PROGRAM_MD)

    for i in range(1, n + 1):
        current = (task_dir / "train.py").read_text()
        history = records[1:]
        user = build_user_message(task, current, best_val, history, sigma_val)
        # Narrow retry on transient API errors (rate limit, overload, 5xx). Any
        # other exception surfaces. This is NOT a generic try/except hiding bugs;
        # it is a targeted handler for anthropic's documented transient failures.
        resp = None
        for attempt in range(5):
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS_PROPOSAL,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                break
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                sleep = 3 * (attempt + 1)
                print(
                    f"[{task}][i={i:03d}] anthropic transient {type(e).__name__}; sleeping {sleep}s"
                )
                time.sleep(sleep)
        assert resp is not None, f"anthropic retries exhausted on {task} i={i}"
        raw = "".join(block.text for block in resp.content if getattr(block, "text", None))
        desc, reflection, new_src = parse_proposal(raw)
        ok, reason = validate_train_py(new_src)
        if not ok:
            records.append(
                {
                    "i": i,
                    "desc": desc,
                    "reflection": reflection,
                    "train_err": float("nan"),
                    "val_err": float("nan"),
                    "test_err": float("nan"),
                    "delta_train": float("nan"),
                    "delta_val": float("nan"),
                    "status": f"error:{reason[:60]}",
                    "train_py_hash": content_hash(new_src),
                }
            )
            records_path.write_text(json.dumps(records, indent=2))
            with proposals_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "i": i,
                            "desc": desc,
                            "reflection": reflection,
                            "new_train_py": new_src,
                            "raw_llm": raw,
                        }
                    )
                    + "\n"
                )
            print(f"[{task}][i={i:03d}] ERR   ({reason[:60]}) {desc[:60]}")
            continue

        prev_src = current
        (task_dir / "train.py").write_text(new_src)
        try:
            ev = eval_current(task, task_dir, log_path, cache)
            cache_path.write_text(json.dumps(cache, indent=2))
        except (RuntimeError, subprocess.TimeoutExpired, AssertionError) as e:
            (task_dir / "train.py").write_text(prev_src)
            records.append(
                {
                    "i": i,
                    "desc": desc,
                    "reflection": reflection,
                    "train_err": float("nan"),
                    "val_err": float("nan"),
                    "test_err": float("nan"),
                    "delta_train": float("nan"),
                    "delta_val": float("nan"),
                    "status": f"runtime_error:{str(e)[:80]}",
                    "train_py_hash": content_hash(new_src),
                }
            )
            records_path.write_text(json.dumps(records, indent=2))
            with proposals_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "i": i,
                            "desc": desc,
                            "reflection": reflection,
                            "new_train_py": new_src,
                            "raw_llm": raw,
                        }
                    )
                    + "\n"
                )
            print(f"[{task}][i={i:03d}] RUN-ERR {desc[:60]}  {str(e)[:60]}")
            continue

        if ev["val_err"] < best_val:
            dv = ev["val_err"] - best_val
            dt = ev["train_err"] - best_train
            best_val = ev["val_err"]
            best_train = ev["train_err"]
            status = "keep"
        else:
            dv = float("nan")
            dt = float("nan")
            status = "discard"
            (task_dir / "train.py").write_text(prev_src)

        records.append(
            {
                "i": i,
                "desc": desc,
                "reflection": reflection,
                "train_err": ev["train_err"],
                "val_err": ev["val_err"],
                "test_err": ev["test_err"],
                "delta_train": dt,
                "delta_val": dv,
                "status": status,
                "train_py_hash": ev["train_py_hash"],
            }
        )
        records_path.write_text(json.dumps(records, indent=2))
        with proposals_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "i": i,
                        "desc": desc,
                        "reflection": reflection,
                        "new_train_py": new_src,
                        "raw_llm": raw,
                    }
                )
                + "\n"
            )
        print(
            f"[{task}][i={i:03d}] {status:7s} val={ev['val_err']:.4f} best={best_val:.4f} "
            f"{desc[:80]}"
        )

    return task, n


def _worker(task: str, n: int, sigma_val: float) -> tuple[str, float]:
    t0 = time.time()
    run_one_task(task, n, sigma_val)
    return task, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()

    sigma_path = REPO / "results" / "_sigma" / "sigma.json"
    assert sigma_path.exists(), "run estimate_sigma.py first"
    sigma_data = json.loads(sigma_path.read_text())

    tasks = args.tasks or TASK_NAMES
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_worker, t, args.n, sigma_data["per_task"][t]["bootstrap_val_std"]): t
            for t in tasks
        }
        done = 0
        for f in as_completed(futs):
            task, dt = f.result()
            done += 1
            print(
                f"[{done:2d}/{len(tasks):2d}] reflection {task:40s} {dt:.1f}s "
                f"elapsed {time.time() - t0:.1f}s"
            )
    print(f"\nreflection total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
