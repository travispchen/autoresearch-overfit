"""Per-task autoresearch loop — one OpenML task, one independent loop.

Each iteration:
  1. Read `results/<task>/train.py` + `program.md` + short history.
  2. LLM proposes ONE mutation, emits full new train.py.
  3. Static checks (AST). Forbidden if it references X_test/y_test or tries
     to reconstruct test data.
  4. Write the new train.py into `results/<task>/`, run `TASK=<name> uv run
     python train.py` from that dir, parse `val_err:`.
  5. Val-gate: strictly lower than current best → keep, else discard (restore
     previous train.py).
  6. Append to `results/<task>/{records.json, results.tsv, proposals.jsonl}`.

Records also carry `test_err`, pulled from the per-task silent test log
after each eval. The agent never sees test_err; the harness reads it from
the silent log for analysis and caches results by content hash for fast
mitigation replays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from prepare import TASK_NAMES, silent_log_path

REPO = Path(__file__).resolve().parent
BASELINE_SRC = (REPO / "train.py").read_text()
PROGRAM_MD = (REPO / "program.md").read_text()

MODEL = "claude-sonnet-4-5"
MAX_TOKENS_PROPOSAL = 8192
SUBPROCESS_TIMEOUT = 300

_DESC_RE = re.compile(r"<description>\s*(.*?)\s*</description>", re.DOTALL)
_TRAIN_RE = re.compile(r"<train_py>\s*(.*?)\s*</train_py>", re.DOTALL)
_TRAIN_OPEN_RE = re.compile(r"<train_py>\s*(.*)", re.DOTALL)
_VAL_ERR_RE = re.compile(r"^val_err:\s*([0-9.]+)\s*$", re.MULTILINE)


@dataclass
class Proposal:
    description: str
    new_train_py: str


def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


SYSTEM_PROMPT_TEMPLATE = """You are an ML researcher running an autoresearch loop on ONE OpenML \
tabular classification task (name = "{task}"). Val_err is the mean misclassification rate on \
the task's val slice; lower is better. The starting `train.py` is an untuned XGBoost with \
library defaults.

Your only job each turn is to propose ONE concrete code change to `train.py` that you expect \
will lower val_err on this task, then emit the full new `train.py` file.

{program_md}

Output format (REQUIRED):

<description>One-line description of the mutation (≤ 20 words, no newlines). If the change \
uses eval_set with early_stopping_rounds, prefix with [es-on-val].</description>
<train_py>
# full new contents of train.py here, verbatim, ready to run via `TASK={task} uv run python train.py`
</train_py>

Do not include any other text outside those two tags. Do not prefix with ```python. The \
train_py tag contents will be written to train.py as-is.
"""


def build_system_prompt(task: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(task=task, program_md=PROGRAM_MD)


def build_user_message(
    task: str, current_src: str, best_val_err: float | None, history: list[dict[str, Any]]
) -> str:
    hist_lines = []
    for h in history[-10:]:
        v = f"{h['val_err']:.4f}" if h["val_err"] == h["val_err"] else "  nan"
        hist_lines.append(f"  [{h['i']:03d}] {h['status']:7s} val={v}  {h['desc']}")
    hist = "\n".join(hist_lines) if hist_lines else "  (none yet)"
    best = f"{best_val_err:.4f}" if best_val_err is not None else "n/a"
    return f"""Task: {task}
Current best val_err: {best}

Recent experiments (last 10):
{hist}

Current train.py:
```python
{current_src}
```

Propose ONE concrete mutation and emit the full new train.py.
"""


def parse_proposal(text: str) -> Proposal:
    dm = _DESC_RE.search(text)
    tm = _TRAIN_RE.search(text)
    assert dm is not None, f"Missing <description> tag:\n{text[:500]}"
    desc = dm.group(1).strip().replace("\t", " ").replace("\n", " ")[:240]
    if tm is not None:
        new_train_py = tm.group(1)
    else:
        # Truncated mid-file — salvage everything after <train_py>.
        om = _TRAIN_OPEN_RE.search(text)
        assert om is not None, f"Missing <train_py> tag:\n{text[:500]}"
        new_train_py = om.group(1)
    new_train_py = re.sub(r"^```[a-zA-Z]*\n", "", new_train_py)
    new_train_py = re.sub(r"\n?```\s*$", "", new_train_py)
    return Proposal(description=desc, new_train_py=new_train_py)


def validate_train_py(src: str) -> tuple[bool, str]:
    """Static checks to block obvious harness violations."""
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    forbidden = {"X_test", "y_test"}
    forbidden_prepare = {"_load_with_test", "silent_log_path"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            return False, f"forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            return False, f"forbidden attr: {node.attr}"
        if isinstance(node, ast.ImportFrom) and node.module == "prepare":
            for alias in node.names:
                if alias.name in forbidden_prepare:
                    return False, f"forbidden import from prepare: {alias.name}"

    if "evaluate_one" not in src:
        return False, "train.py must call evaluate_one"
    if 'TASK = os.environ["TASK"]' not in src and "TASK = os.environ['TASK']" not in src:
        return False, 'TASK = os.environ["TASK"] must be preserved'
    if 'if __name__ == "__main__"' not in src and "if __name__ == '__main__'" not in src:
        return False, "__main__ block required"
    return True, ""


def propose(
    client: anthropic.Anthropic,
    task: str,
    current_src: str,
    best_val_err: float | None,
    history: list[dict[str, Any]],
) -> tuple[Proposal, str]:
    system = build_system_prompt(task)
    user_msg = build_user_message(task, current_src, best_val_err, history)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_PROPOSAL,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = "".join(block.text for block in resp.content if getattr(block, "text", None))
    return parse_proposal(raw), raw


def read_silent_log_tail(log_path: Path) -> dict[str, Any]:
    with log_path.open() as f:
        last = None
        for line in f:
            line = line.strip()
            if line:
                last = line
    assert last is not None
    return json.loads(last)


def run_train(task: str, task_dir: Path) -> float:
    env = os.environ.copy()
    env["TASK"] = task
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "--project", str(REPO), "python", str(task_dir / "train.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"train.py exit={proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
    m = _VAL_ERR_RE.search(proc.stdout)
    if not m:
        raise RuntimeError(f"could not parse val_err from stdout:\n{proc.stdout[-2000:]}")
    return float(m.group(1))


def eval_current_train(
    task: str, task_dir: Path, log_path: Path, cache: dict[str, Any]
) -> dict[str, Any]:
    src = (task_dir / "train.py").read_text()
    h = content_hash(src)
    if h in cache:
        return cache[h]
    val_err = run_train(task, task_dir)
    tail = read_silent_log_tail(log_path)
    assert abs(tail["val_err"] - val_err) < 1e-4, (
        f"val mismatch: stdout={val_err} silent={tail['val_err']}"
    )
    cache[h] = {
        "train_py_hash": h,
        "train_err": tail["train_err"],
        "val_err": tail["val_err"],
        "test_err": tail["test_err"],
    }
    return cache[h]


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def append_proposal(path: Path, i: int, desc: str, new_src: str, raw: str) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"i": i, "desc": desc, "new_train_py": new_src, "raw_llm": raw}) + "\n")


def append_tsv(path: Path, i: int, desc: str, val_err: float | str, status: str) -> None:
    val_str = f"{val_err:.6f}" if isinstance(val_err, float) else str(val_err)
    with path.open("a") as f:
        f.write(f"{i}\t{desc}\t{val_str}\t{status}\n")


def _init_task_dir(task: str, task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "train.py").write_text(BASELINE_SRC)


def run_one_task(task: str, n: int, seed: int = 0, resume: bool = False) -> None:
    assert task in TASK_NAMES, f"unknown task: {task}"
    random.seed(seed)

    task_dir = REPO / "results" / task
    records_path = task_dir / "records.json"
    proposals_path = task_dir / "proposals.jsonl"
    cache_path = task_dir / "eval_cache.json"
    tsv_path = task_dir / "results.tsv"
    log_path = silent_log_path(task)

    _init_task_dir(task, task_dir)

    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    records: list[dict[str, Any]] = []
    if resume and records_path.exists():
        records = json.loads(records_path.read_text())
        # Restore the last-kept train.py into the task dir.
        kept = [r for r in records if r["status"] == "keep"]
        if kept and proposals_path.exists():
            props = [json.loads(line) for line in proposals_path.read_text().splitlines()]
            last_i = kept[-1]["i"]
            matches = [p for p in props if p["i"] == last_i]
            if matches and "new_train_py" in matches[0]:
                (task_dir / "train.py").write_text(matches[0]["new_train_py"])

    best_val: float | None = None
    if not records:
        ev0 = eval_current_train(task, task_dir, log_path, cache)
        save_json(cache, cache_path)
        best_val = ev0["val_err"]
        records.append(
            {
                "i": 0,
                "desc": "baseline",
                "train_err": ev0["train_err"],
                "val_err": ev0["val_err"],
                "test_err": ev0["test_err"],
                "status": "keep",
                "train_py_hash": ev0["train_py_hash"],
            }
        )
        save_json(records, records_path)
        append_proposal(proposals_path, 0, "baseline", (task_dir / "train.py").read_text(), "")
        append_tsv(tsv_path, 0, "baseline", best_val, "keep")
        print(f"[{task}][i=000] keep  baseline  val={best_val:.4f}")
    else:
        best_val = min(
            r["val_err"] for r in records if r["status"] == "keep" and r["val_err"] == r["val_err"]
        )

    client = anthropic.Anthropic()
    start_i = (max(r["i"] for r in records) + 1) if records else 1

    for i in range(start_i, n + 1):
        history = records[1:]
        current = (task_dir / "train.py").read_text()
        t0 = time.time()
        proposal, raw = propose(client, task, current, best_val, history)
        t_llm = time.time() - t0

        ok, reason = validate_train_py(proposal.new_train_py)
        if not ok:
            records.append(
                {
                    "i": i,
                    "desc": proposal.description,
                    "train_err": float("nan"),
                    "val_err": float("nan"),
                    "test_err": float("nan"),
                    "status": f"error:{reason[:60]}",
                    "train_py_hash": content_hash(proposal.new_train_py),
                }
            )
            save_json(records, records_path)
            append_proposal(proposals_path, i, proposal.description, proposal.new_train_py, raw)
            append_tsv(tsv_path, i, proposal.description, "nan", "error")
            print(f"[{task}][i={i:03d}] ERR   {proposal.description[:60]}  ({reason[:60]})")
            continue

        prev = current
        (task_dir / "train.py").write_text(proposal.new_train_py)
        t1 = time.time()
        try:
            ev = eval_current_train(task, task_dir, log_path, cache)
            save_json(cache, cache_path)
        except (RuntimeError, subprocess.TimeoutExpired, AssertionError) as e:
            (task_dir / "train.py").write_text(prev)
            records.append(
                {
                    "i": i,
                    "desc": proposal.description,
                    "train_err": float("nan"),
                    "val_err": float("nan"),
                    "test_err": float("nan"),
                    "status": f"runtime_error:{str(e)[:120]}",
                    "train_py_hash": content_hash(proposal.new_train_py),
                }
            )
            save_json(records, records_path)
            append_proposal(proposals_path, i, proposal.description, proposal.new_train_py, raw)
            append_tsv(tsv_path, i, proposal.description, "nan", "runtime_error")
            print(f"[{task}][i={i:03d}] RUN-ERR {proposal.description[:60]}  {str(e)[:60]}")
            continue
        t_eval = time.time() - t1

        val_err = ev["val_err"]
        if val_err < best_val:
            status = "keep"
            best_val = val_err
        else:
            status = "discard"
            (task_dir / "train.py").write_text(prev)

        records.append(
            {
                "i": i,
                "desc": proposal.description,
                "train_err": ev["train_err"],
                "val_err": val_err,
                "test_err": ev["test_err"],
                "status": status,
                "train_py_hash": ev["train_py_hash"],
            }
        )
        save_json(records, records_path)
        append_proposal(proposals_path, i, proposal.description, proposal.new_train_py, raw)
        append_tsv(tsv_path, i, proposal.description, val_err, status)
        print(
            f"[{task}][i={i:03d}] {status:7s} val={val_err:.4f} best={best_val:.4f}  "
            f"t_llm={t_llm:.1f}s t_eval={t_eval:.1f}s  {proposal.description[:80]}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="OpenML task name")
    ap.add_argument("--n", type=int, default=25, help="number of experiments (incl. baseline)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_one_task(args.task, args.n, seed=args.seed, resume=args.resume)


if __name__ == "__main__":
    main()
