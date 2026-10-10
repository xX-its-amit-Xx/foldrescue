# AGENTS.md — working rules for this repository

Guidance for any coding agent editing `foldrescue`. These are not
style preferences; each one exists because violating it produces output that
looks correct and is not.

## 1. The resource guard is not optional

`foldrescue.utils.check_resources()` gates every allocation-heavy path. The
contract:

```python
plan = check_resources(n_rows, n_cols, dtype=np.float64, n_jobs=None,
                       memory_fraction=0.5, label="what this is for")
# -> ResourcePlan(dtype, chunk_rows, n_jobs, n_chunks, estimated_bytes, ...)
# -> raises ResourceRefusal when no permitted mitigation fits
```

It (1) reads **available** RAM and CPU count via `psutil`, (2) estimates the
dense footprint as `rows × cols × itemsize × working_copies`, (3) if that
exceeds `memory_fraction` (default **0.5**) of available RAM, mitigates in a
**fixed order** — downcast float64→float32, then chunk along rows, then cap
`n_jobs` at `nproc − 1` — and (4) otherwise raises `ResourceRefusal` naming the
numbers and the knobs that would make it fit.

**Rules:**

- Call it **before** allocating, never after.
- **Honour the plan you receive.** Use `plan.dtype`, iterate with
  `iter_chunks(n_rows, plan)`, pass `plan.n_jobs` to the estimator. Consulting
  the guard and then allocating `float64` in one shot is worse than not calling
  it, because the logs claim it was checked.
- **Never raise `memory_fraction` to make a refusal go away.** The refusal is
  the feature. If a legitimate workload needs more, reduce feature width or
  chunk harder.
- A `ResourceRefusal` is a **successful** outcome. Do not catch it to continue
  anyway. The CLI surfaces it with exit code 2, distinct from a crash.
- Why 0.5 and not 0.9: the estimate is a lower bound. Python objects, BLAS
  scratch and the interpreter live outside it.
- Keep `tests/test_resource_guard.py` passing. The refusal tests matter most.

Embedding extraction additionally **must** stream and checkpoint — batch,
flush, record the id in the manifest. Never buffer a whole extraction in RAM
and write once; the step most likely to be killed halfway is exactly this one.

## 2. Synthetic data must be labelled everywhere it surfaces

The demonstration cohort is generated. It is a test fixture at benchmark scale,
**not a dataset**. Any artefact derived from it carries that fact:

- `Cohort.synthetic is True` and `Cohort.label_banner()` says so;
- the CLI prints a loud stderr banner and writes `# SYNTHETIC TRAINING DATA`
  into the CSV header;
- notebook figures are watermarked; result CSVs are prefixed `DEMO_synthetic_`.

**Never** name a synthetic output as though it were a real prediction (no
`glb1_shortlist.csv`), never quote its metrics as findings, and never remove a
watermark to tidy a figure. A committed CSV outlives the notebook caveat that
produced it.

## 3. Calibration is fitted on the calibration split. Only.

Every split is three-way (`train` / `calib` / `test`). Fit calibrators,
Venn–Abers, conformal quantiles **and the decision threshold** on `calib`.
A temperature or threshold fitted on `test` is a fit statistic, not a
measurement, and `fit_calibrator` labels in-sample reports `(in-sample)` for
exactly this reason — do not strip that label.

Do not threshold at 0.5. At ~30% prevalence a calibrated model rarely exceeds
it, yielding zero positive predictions and an MCC of exactly 0. Use
`evaluate.choose_threshold` on `calib`.

## 4. Do not let the model cheat

- `Cohort.oracle_p` (the generative probability) is held **off** `pairs` so it
  cannot be swept into a feature matrix. Keep it that way.
- Fit the applicability domain on **training rows only**. Fitting it on the
  whole cohort certifies queries the model never saw — the subtle form of
  test-set leakage.
- Grouped splits must not leak: no gene across `gene_holdout`, no compound
  across `compound_holdout`. Tests assert this.

## 5. Feature blocks stay separable

`stability`, `structural`, `ligand`, `interaction`. The ablation notebook is
only meaningful if removing a block removes exactly that block. When adding a
feature, put it in a block and register it in `block_of`.

Dropping the ligand block is an **ablation, not a configuration**: without it
the model cannot distinguish compounds and cannot answer the pair question at
all. Say so wherever it is offered.

## 6. Tier 3 must stay architecturally identical to Tier 1

`Tier3StructureAware` subclasses `Tier1GBM` on purpose, so the difference
between their numbers is attributable to the pocket-occupancy features and
nothing else. Do not "improve" Tier 3 with a different architecture — that
silently converts a clean experiment into an uninterpretable one.

## 7. Keep the encoders frozen by default

Tier 2's encoders have 10⁷–10⁸ parameters against ~10³ labelled pairs.
`unfreeze_encoders=True` must keep its loud warning. Do not quietly raise the
default capacity of the fusion head either.

## 8. Report uncertainty that means something

`ci_low`/`ci_high` are **Venn–Abers** intervals. Do not revert them to
split-conformal regression on `|y − p|`: with binary labels that yields q ≈ 0.9
and every interval collapses to `[0, 1]` — valid and useless. This repository
shipped that for one iteration; the README records it. Conformal is retained
for **label sets**, which is its meaningful output here.

## 9. The CLI refuses to rank extrapolations

Out-of-domain pairs are withheld unless `--include-ood`, and even then are
appended *after* every in-domain row, never co-sorted. Do not "simplify" this
into one sorted list. A ranked CSV is read from the top and an extrapolation's
number looks identical to a supported one.

## 10. Attribution answers a different question from promiscuity

SHAP and permutation importance explain **the model**. They do not establish
that it learned biology. Keep `interrogate_promiscuity` — the compound
base-rate baseline and the within-variant shuffle — as the load-bearing check,
and keep its verdict phrased as a sentence rather than a bare number.

## 11. Honesty rules that override tidiness

- Never present a metric computed on synthetic data as a result.
- Never delete a warning to make output cleaner.
- If a test fails, report the failure. Do not weaken the assertion to pass.
  (Two tests in this repo were *wrong* and were fixed — but only after
  establishing that the code was right. That order matters.)
- If a mitigation changed the numbers (downcast, chunking), say so in the log.
- Prefer refusing to guessing: `AmenabenchAdapter` raises an error naming the
  paths it checked rather than silently falling back to synthetic data.

## Note for agents other than Claude Code

This file is kept byte-for-byte equivalent in substance to `CLAUDE.md`, which
Claude Code reads automatically. If you edit one, edit both — a divergence
means two agents working on this repository follow different rules, which is
how the honesty rules above get quietly dropped.

## Layout

```
foldrescue/
  utils.py        check_resources, ResourcePlan, ResourceRefusal, logging
  data.py         BenchmarkAdapter, Cohort, synthetic generator, splits
  features.py     the four feature blocks, assembled under the guard
  models.py       Tier1GBM, Tier2Fusion, Tier3StructureAware
  calibration.py  temperature, isotonic, Brier/ECE, conformal, Venn-Abers
  domain.py       two-axis applicability domain
  attribution.py  permutation, SHAP, promiscuity interrogation
  embeddings.py   streaming extraction + resumable checkpoint cache
  evaluate.py     metrics, threshold selection, multi-seed aggregation
  cli.py          foldrescue predict / resources
```

## Checks before committing

```bash
ruff check foldrescue tests
pytest -q
```
