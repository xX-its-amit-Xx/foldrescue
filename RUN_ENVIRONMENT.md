# Run environment

Provenance for every executed notebook, figure and CSV committed under
`results/`. Recorded automatically at execution time, not transcribed by hand.

## Where this ran

| | |
|---|---|
| Platform | Linux-6.18.44-fc-v114-x86_64-with-glibc2.39 |
| Python | 3.13.16 (CPython) |
| CPU cores | 4 logical |
| RAM | 15.7 GiB total |
| GPU | **none** — all notebooks executed on CPU |
| Executed | 2026-10-10 |
| Executor | `nbclient`, `allow_errors=False` (any exception fails the run) |

The spec suggested Kaggle (GPU) or Codespaces. This ran in a Claude Code cloud
container instead. Nothing here needs a GPU: the two tree tiers are CPU-only by
construction, and the Tier 2 fusion head is small enough that 80 epochs over
~900 training pairs takes a few seconds on four cores. A GPU would change the
runtimes below and nothing else.

## Package versions

| Package | Version |
|---|---|
| `numpy` | 2.5.3 |
| `pandas` | 3.0.5 |
| `scipy` | 1.18.1 |
| `scikit-learn` | 1.9.1 |
| `lightgbm` | 4.7.0 |
| `rdkit` | 2026.9.1 |
| `shap` | 0.53.0 |
| `psutil` | 7.2.2 |
| `matplotlib` | 3.11.2 |
| `torch` | 2.14.1 |
| `transformers` | 5.19.0 |
| `nbformat` | 5.11.1 |
| `nbclient` | 0.11.0 |
| `pytest` | 9.1.1 |
| `ruff` | 0.17.0 |

## Notebook runtimes

Wall-clock, measured by the executor. Totals exclude notebook build time.

| Notebook | Runtime |
|---|---|
| `cookbook/01_train_all_tiers.ipynb` | 67.8 s |
| `cookbook/02_calibration_study.ipynb` | 6.3 s |
| `cookbook/03_glb1_prospective.ipynb` | 5.1 s |
| `cookbook/04_ablation.ipynb` | 25.6 s |
| **total** | **104.8 s** |

`01` dominates because it is 3 tiers x 4 splits x 5 seeds = 60 model fits, 20 of
them Tier 2.

## Two environment limitations that affect what the results mean

### 1. Tier 2 ran on SURROGATE embeddings, not ESM-2 / ChemBERTa

This container's egress proxy allowlists PyPI but **not** `huggingface.co`:

```
$ curl https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/config.json
curl: (56) CONNECT tunnel failed, response 403
```

So pretrained weights could not be fetched, and
`foldrescue.embeddings` fell back to its documented surrogate backend —
hashed k-mer counts for the protein window, a seeded random projection of
Morgan bits for the compound. Both are deterministic and reproducible; neither
is a language-model representation.

**Consequence:** the Tier 2 numbers in `01_train_all_tiers.ipynb` exercise the
cross-attention *architecture* and say **nothing** about ESM-2 or ChemBERTa.
Tier 2 underperforming the tree tiers here is therefore only weak evidence
about representation fusion in general — it is partly a statement about the
surrogate. On a host where `huggingface.co` is reachable, pass
`protein_backend="esm2", compound_backend="chemberta"` to
`build_embedding_matrix` and the real encoders are used with no other change.
The fallback is always announced at WARNING level and recorded in the
provenance dict returned alongside the matrix.

### 2. All results are on SYNTHETIC data

`amenabench`, the curated benchmark this package is designed to consume,
contains only a `LICENSE` file at the time of writing — there is no data in it.
Every notebook therefore runs on `foldrescue.data.SyntheticCohortAdapter`.

Every artefact is labelled accordingly: figures carry a `SYNTHETIC` watermark,
result CSVs are prefixed `DEMO_synthetic_`, and the holdout shortlist is
**not** named after the gene it holds out, so the file cannot be mistaken for a
prediction once separated from its notebook.

## Reproducing

```bash
pip install -e ".[all]"
cd cookbook
for nb in 01_train_all_tiers 02_calibration_study 03_glb1_prospective 04_ablation; do
    jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"
done
```

All randomness derives from explicit seeds (`SEEDS = [0, 1, 2, 3, 4]`), so a
rerun on the same package versions reproduces the committed numbers exactly.
Different LightGBM or scikit-learn versions may shift the low-order digits;
the qualitative conclusions are stable across the seed range, which is what the
error bars are reporting.

## Verification at time of commit

```
pytest -q        ->  120 passed, 0 skipped
ruff check       ->  All checks passed
```
