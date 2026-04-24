# autoresearch-overfit

A small harness that measures how much val-gated LLM autoresearch overfits the validation set,
and tests several fixes. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
and [issue #131](https://github.com/karpathy/autoresearch/issues/131) on random-seed engineering.

The full study is in **[WRITEUP.md](./WRITEUP.md)** — plots, numbers, and analysis.

**TL;DR.** Claude Sonnet 4.5 edits `train.py` 25 times per task across 15 OpenML tabular
classification tasks, with a strict val-gate as the accept/reject rule. The starting baseline is
an untuned `XGBClassifier`. Across 375 LLM proposals, mean Δval is +0.019 (paired Wilcoxon
p = 0.001) but mean Δtest is essentially zero (-0.0006). On 7 of the 12 tasks with any movement,
val improves while test gets *worse*. Requiring `Δval > 1·σ_val` at the gate (effect-size gating)
closes the val−test gap to +0.005 (p = 0.004) at the cost of fewer accepted experiments.

## Repo layout

```
.
├── prepare.py                    # data prep + silent evaluate_one()  (HARNESS — not agent-editable)
├── train.py                      # per-task agent-editable XGBoost baseline
├── program.md                    # per-task agent rulebook
├── run_autoresearch.py           # per-task driver: LLM proposer + val-gate + trajectory
├── launch.py                     # fans out 15 per-task loops in parallel
├── scripts/
│   ├── estimate_sigma.py         # per-task bootstrap val-slice σ (noise floor)
│   ├── replay_mitigations.py     # replay each task's proposal stream through each gate
│   ├── run_reflection.py         # proposer-side "reflect on overfit" fresh LLM loop
│   ├── noise_replay.py           # bonus: pure-seed-swap lower bound (§7)
│   ├── populate_val_per_sample.py # backfill per-sample val errors needed by rotating_val
│   ├── plot_overfit_per_task.py  # 3×5 per-task summary grid
│   ├── plot_roofline.py          # aggregate plots used in the writeup
│   ├── plot_comparison.py        # per-task / per-gate summary table
│   └── stats_significance.py     # paired Wilcoxon tests + bootstrap CIs
├── results/                      # cached per-task records, per-gate replays, summary plots/JSON
└── WRITEUP.md
```

## Quickstart

```bash
uv sync
uv run python prepare.py             # one-time fetch + cache of all 15 OpenML tasks
TASK=credit-g uv run python train.py # sanity-check one task; prints `val_err: …`

# Baseline loop — needs ANTHROPIC_API_KEY
uv run python launch.py --n 25       # 15 parallel per-task loops, ~5 min wall

# Mitigation replays — pure replays, no API calls
uv run python scripts/estimate_sigma.py
uv run python scripts/replay_mitigations.py
uv run python scripts/run_reflection.py --n 25     # fresh LLM loop, ~5 min, needs ANTHROPIC_API_KEY
uv run python scripts/noise_replay.py --n 50       # ~2 min

# Plots and significance
uv run python scripts/plot_overfit_per_task.py
uv run python scripts/plot_roofline.py
uv run python scripts/plot_comparison.py
uv run python scripts/stats_significance.py
```

`results/` is checked in so the writeup is reproducible end-to-end without re-running any LLM
calls. Re-running `replay_mitigations.py`, `plot_*.py`, and `stats_significance.py` regenerates
every number and figure in the writeup from the cached per-proposal records.

## Harness invariants

- `prepare.py` owns `X_test` / `y_test`. They do not appear on the agent-visible `Dataset`
  object and are not returned from `evaluate_one()`. Test errors are written to a private log
  at `~/.cache/autoresearch-toy/silent_test_log_<task>.jsonl` that no proposer prompt, gate,
  or program rule reads during the loop.
- Every proposed `train.py` is AST-validated before running. References to `X_test`, `y_test`,
  `_load_with_test`, or `silent_log_path` are rejected.
- Parallelism lives at the outer 15-task level (one worker per task) with `OMP_NUM_THREADS=1`
  and XGBoost `n_jobs=1` so each worker uses a single thread.
- Every eval is content-hash cached inside `results/<task>/eval_cache.json`. Identical
  `train.py` proposals are deduplicated.

## Datasets

15 OpenML tabular classification tasks, version 1 of each:

```
credit-g, diabetes, kc1, phoneme, vehicle, wilt, spambase,
mushroom, blood-transfusion-service-center, banknote-authentication,
splice, segment, satimage, mfeat-factors, cmc
```

Stratified 65/10/25 train/val/test split, fixed seed. Numeric NaNs filled with train medians,
categoricals one-hot encoded with the train vocabulary. No standard scaler — XGBoost is
scale-invariant. `mushroom` and `banknote-authentication` are near-zero-error null controls.

## Why XGBoost?

The baseline matters. If the starting point is already strong on tabular data, real
improvements are small and rare, and most of the val-gate's accepted "wins" must either be
(a) genuine but tiny gains or (b) noise-mining on the val slice. That is exactly the regime
where adaptive holdout reuse bites. Untuned XGBoost is hard to beat on these tasks, which is
the point.

## XGBoost-specific leakage caveat

`XGBClassifier` supports `eval_set=[(X_val, y_val)]` with `early_stopping_rounds=K`, which
makes the val slice both the gate AND the early-stopping oracle — a compounded val-leak worse
than the val-gate leak this study is measuring. Any agent proposal using it must prefix the
description with `[es-on-val]` (~0 of 48 kept baseline proposals do, so it's negligible in
practice for our budget).
