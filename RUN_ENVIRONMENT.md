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
| Network | unrestricted egress (see below) |
| Executed | 2026-10-10 |
| Executor | `nbclient`, `allow_errors=False` (any exception fails the run) |

The spec suggested Kaggle (GPU) or Codespaces. This ran in a Claude Code cloud
container instead. Nothing here needs a GPU: the two tree tiers are CPU-only by
construction, ESM-2 t6 (7.5M parameters) and ChemBERTa (44.1M) are run once as
frozen feature extractors with the results cached to disk, and the fusion head
is small enough to train in seconds on four cores.

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

| Notebook | Runtime |
|---|---|
| `cookbook/01_train_all_tiers.ipynb` | 81.1 s |
| `cookbook/02_calibration_study.ipynb` | 6.4 s |
| `cookbook/03_glb1_prospective.ipynb` | 4.6 s |
| `cookbook/04_ablation.ipynb` | 26.7 s |
| **total** | **118.8 s** |

`01` dominates: 3 tiers x 4 splits x 5 seeds = 60 model fits, 20 of them
Tier 2. Embedding extraction is cached and resumable, so only the first seed
pays for it.

## Encoders: real, and cached

Tier 2 uses the real pretrained checkpoints:

| Side | Checkpoint | Width | Parameters |
|---|---|---|---|
| Protein | `facebook/esm2_t6_8M_UR50D` | 320 | 7.5M |
| Compound | `seyonec/ChemBERTa-zinc-base-v1` | 768 | 44.1M |

Both are frozen. Extraction streams in batches and checkpoints to
`.foldrescue_cache/embeddings/`, keyed by a fingerprint of the extraction
configuration so a cache built with one model can never be resumed under
another.

An earlier run of this repository used a documented `hashed` surrogate backend,
because the container's egress proxy did not allow `huggingface.co`. That
restriction has since been lifted and the real encoders are now the default.
The surrogate remains available for offline use and announces itself at WARNING
level whenever active.

Protein sequences come from `foldrescue/assets/sequences.json` (6 KB, retrieved
from the UniProt REST API and committed), so the cohort is reproducible with no
network at all.

## The limitation that matters most

**All results are on SYNTHETIC data, and Tier 2 is structurally capped by it.**

`amenabench`, the curated benchmark this package is designed to consume,
contains only a `LICENSE` file at the time of writing. Every notebook therefore
runs on `foldrescue.data.SyntheticCohortAdapter`.

That cohort's protein side is real — real UniProt sequences, real positions,
real wild-type residues. But `ddg_pred`, `rsa`, `conservation` and the pocket
geometry are drawn from distributions, not computed from structure. Since the
generative pair term crosses compound descriptors against exactly those
columns, and they correlate with sequence position at r ~ 0.02, **the
variant-side signal is not recoverable from sequence by any encoder.** Tier 2
therefore underperforms here for a reason that says nothing about
representation fusion. Notebook 01 measures this rather than asserting it.

Every artefact is labelled: figures carry a `SYNTHETIC` watermark, result CSVs
are prefixed `DEMO_synthetic_`, and the holdout shortlist is **not** named after
the gene it holds out.

## Reproducing

```bash
pip install -e ".[all]"
cd cookbook
for nb in 01_train_all_tiers 02_calibration_study 03_glb1_prospective 04_ablation; do
    jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"
done
```

All randomness derives from explicit seeds (`SEEDS = [0, 1, 2, 3, 4]`). The
first run downloads the two checkpoints from Hugging Face; subsequent runs use
the embedding cache.

## Verification at time of commit

```
pytest -q        ->  164 passed, 0 skipped
ruff check       ->  All checks passed
```
