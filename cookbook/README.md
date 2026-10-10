# Cookbook

Four executed notebooks. Outputs are committed, figures land in
`results/figures/`, and `../RUN_ENVIRONMENT.md` records versions and runtimes.

| Notebook | Question it answers |
|---|---|
| `01_train_all_tiers.ipynb` | How do the three tiers compare across all four splits, with error bars over 5 seeds — and against the Bayes ceiling? |
| `02_calibration_study.ipynb` | Does calibration reduce ECE on held-out data, and what would uncalibrated confidence have cost in wasted assays? |
| `03_glb1_prospective.ipynb` | What happens when you ask for a prospective prediction on a gene the model has never seen? |
| `04_ablation.ipynb` | What does each feature block contribute — and did the model learn pair-specific signal or just compound promiscuity? |

## Everything here runs on synthetic data

`amenabench` contains only a `LICENSE` at the time of writing, so all four
notebooks use `foldrescue.data.SyntheticCohortAdapter`. Figures are
watermarked, result CSVs are prefixed `DEMO_synthetic_`, and the holdout
shortlist is deliberately **not** named after the gene it holds out.

These notebooks demonstrate that the pipeline runs and that the diagnostics
behave as designed against a known ground truth. They are not findings about
chaperone biology.

## Re-executing

```bash
pip install -e "..[all]"
cd cookbook
jupyter nbconvert --to notebook --execute --inplace 01_train_all_tiers.ipynb
```

Seeds are explicit, so a rerun on the same versions reproduces the committed
numbers. `_viz.py` holds the shared plotting style and the provenance of the
CVD-validated palette.
