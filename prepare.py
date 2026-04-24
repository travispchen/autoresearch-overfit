"""Per-task data prep + silent evaluate_one() API.

The harness owns X_test/y_test. The agent (and `train.py`) never sees them.
Each `evaluate_one` call computes val_err AND test_err, silently logs
test_err to a JSONL file, and returns ONLY val info.

Strict invariant: no test info ever flows out of `evaluate_one`'s return
value. Audit before every run.

Tasks: 15 OpenML classification datasets, val_err = misclassification rate.
"""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
import zlib

# Clamp threaded math libraries BEFORE numpy/xgboost are imported anywhere in
# this Python process. Parallelism lives at the outer per-task level (one
# worker process per task), so each process should use exactly one thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

SEED = 20260422
VAL_FRAC = 0.10
TEST_FRAC = 0.25
CACHE_DIR = Path(os.path.expanduser("~/.cache/autoresearch-toy"))
DATA_DIR = CACHE_DIR / "data"

OPENML_TASKS: list[tuple[str, int]] = [
    ("credit-g", 1),
    ("diabetes", 1),
    ("kc1", 1),
    ("phoneme", 1),
    ("vehicle", 1),
    ("wilt", 1),
    ("spambase", 1),
    ("mushroom", 1),
    ("blood-transfusion-service-center", 1),
    ("banknote-authentication", 1),
    ("splice", 1),
    ("segment", 1),
    ("satimage", 1),
    ("mfeat-factors", 1),
    ("cmc", 1),
]
TASK_NAMES: list[str] = [n for n, _ in OPENML_TASKS]


@dataclass
class Dataset:
    """Agent-visible dataset. Deliberately does NOT carry X_test/y_test."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    n_classes: int
    n_features: int


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.npz"


def silent_log_path(task_name: str) -> Path:
    return CACHE_DIR / f"silent_test_log_{task_name}.jsonl"


def _prep_openml(name: str, version: int) -> None:
    """Fetch, split, preprocess one OpenML classification task, cache to npz."""
    path = _cache_path(name)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    bunch = fetch_openml(name=name, version=version, as_frame=True, cache=True, parser="auto")
    X = bunch.data
    y = bunch.target

    y_codes, _ = pd.factorize(y, sort=True)
    assert (y_codes >= 0).all(), f"{name}: NaN labels not supported"
    n_classes = int(y_codes.max() + 1)

    idx_all = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=TEST_FRAC, random_state=SEED, stratify=y_codes
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_FRAC / (1.0 - TEST_FRAC),
        random_state=SEED,
        stratify=y_codes[idx_trainval],
    )

    X_train = X.iloc[idx_train].copy()
    X_val = X.iloc[idx_val].copy()
    X_test = X.iloc[idx_test].copy()

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    if num_cols:
        imputer = SimpleImputer(strategy="median")
        X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
        X_val[num_cols] = imputer.transform(X_val[num_cols])
        X_test[num_cols] = imputer.transform(X_test[num_cols])

    if cat_cols:
        X_train_cat = pd.get_dummies(X_train[cat_cols], dummy_na=False)
        vocab = X_train_cat.columns
        X_val_cat = pd.get_dummies(X_val[cat_cols], dummy_na=False).reindex(
            columns=vocab, fill_value=0
        )
        X_test_cat = pd.get_dummies(X_test[cat_cols], dummy_na=False).reindex(
            columns=vocab, fill_value=0
        )
        X_train = pd.concat([X_train[num_cols], X_train_cat], axis=1)
        X_val = pd.concat([X_val[num_cols], X_val_cat], axis=1)
        X_test = pd.concat([X_test[num_cols], X_test_cat], axis=1)
    else:
        X_train = X_train[num_cols]
        X_val = X_val[num_cols]
        X_test = X_test[num_cols]

    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns), (
        f"{name}: column alignment failed"
    )

    np.savez(
        path,
        X_train=X_train.to_numpy(dtype=np.float32),
        y_train=y_codes[idx_train].astype(np.int64),
        X_val=X_val.to_numpy(dtype=np.float32),
        y_val=y_codes[idx_val].astype(np.int64),
        X_test=X_test.to_numpy(dtype=np.float32),
        y_test=y_codes[idx_test].astype(np.int64),
        n_classes=np.array([n_classes], dtype=np.int64),
    )


def ensure_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, version in OPENML_TASKS:
        _prep_openml(name, version)


def load_dataset(task_name: str) -> Dataset:
    """Agent-visible data for one task. Train + val only."""
    ensure_cache()
    d = np.load(_cache_path(task_name), allow_pickle=False)
    return Dataset(
        name=task_name,
        X_train=d["X_train"],
        y_train=d["y_train"],
        X_val=d["X_val"],
        y_val=d["y_val"],
        n_classes=int(d["n_classes"][0]),
        n_features=d["X_train"].shape[1],
    )


def _load_with_test(task_name: str) -> tuple[Dataset, np.ndarray, np.ndarray]:
    """Harness-only: load train+val+test. Never called by agent code."""
    ensure_cache()
    d = np.load(_cache_path(task_name), allow_pickle=False)
    ds = Dataset(
        name=task_name,
        X_train=d["X_train"],
        y_train=d["y_train"],
        X_val=d["X_val"],
        y_val=d["y_val"],
        n_classes=int(d["n_classes"][0]),
        n_features=d["X_train"].shape[1],
    )
    return ds, d["X_test"], d["y_test"]


def _git_hash() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    return out.stdout.strip() if out.returncode == 0 else "nogit"


def _encode_array(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(a.tobytes(), 6)).decode("ascii")


def decode_val_per_sample(s: str, n: int) -> np.ndarray:
    """Inverse of the encoding used in the silent log. Used by gate replays."""
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype=np.float32).reshape(n)


def evaluate_one(task_name: str, train_one_fn) -> dict:
    """Run train_one_fn(ds) for this task. Compute val_err AND test_err,
    silently log test_err AND per-sample val errors to the per-task silent
    log, return ONLY val_err and train_seconds to the caller.

    val_err is the mean misclassification rate. Per-sample val errors
    (0/1 per sample) are stored so the rotating-val gate can evaluate any
    experiment on any disjoint fold of the val slice after the fact.
    """
    ds, X_test, y_test = _load_with_test(task_name)
    t0 = time.time()
    predict = train_one_fn(ds)
    dt = time.time() - t0
    train_pred = predict(ds.X_train)
    val_pred = predict(ds.X_val)
    test_pred = predict(X_test)
    train_err = float((train_pred != ds.y_train).mean())
    val_err = float((val_pred != ds.y_val).mean())
    test_err = float((test_pred != y_test).mean())
    val_per_sample = (val_pred != ds.y_val).astype(np.float32)
    caller = Path(sys.argv[0]).resolve()
    assert caller.exists(), f"caller not found: {caller}"
    train_py_hash = hashlib.sha256(caller.read_bytes()).hexdigest()[:16]

    rec = {
        "t": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": task_name,
        "git_hash": _git_hash(),
        "train_py_hash": train_py_hash,
        "train_err": train_err,
        "val_err": val_err,
        "test_err": test_err,
        "train_seconds": dt,
        "n_val": int(val_per_sample.shape[0]),
        "val_per_sample_b64": _encode_array(val_per_sample),
    }
    log_path = silent_log_path(task_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")

    # AGENT-VISIBLE RETURN: no test info.
    return {"val_err": val_err, "train_seconds": dt}


if __name__ == "__main__":
    ensure_cache()
    for name in TASK_NAMES:
        ds = load_dataset(name)
        print(
            f"{ds.name:40s} train={ds.X_train.shape} val={ds.X_val.shape} "
            f"nc={ds.n_classes} nf={ds.n_features}"
        )
