# Results

Committed outputs from `../cookbook/`. See `../RUN_ENVIRONMENT.md` for versions
and runtimes.

## The `DEMO_synthetic_` prefix is load-bearing

Every file here is produced from generated data by
`foldrescue.data.SyntheticCohortAdapter`. The prefix exists because **a
committed CSV outlives the notebook caveat that produced it**: a file named
`glb1_shortlist.csv` sitting in a repository will eventually be read as a claim
about GLB1, however carefully the notebook hedged.

Do not rename these files, and do not cite their numbers.

| File | Contents |
|---|---|
| `DEMO_synthetic_tier_comparison.csv` | PR-AUC / MCC / Brier per split and tier, mean ± SD over 5 seeds, with the Bayes ceiling |
| `DEMO_synthetic_calibration_report.csv` | ECE and Brier before/after temperature and isotonic calibration |
| `DEMO_synthetic_calibration_cost.csv` | Promised vs delivered hit rate in the top-k, i.e. the cost of overconfidence in assays |
| `DEMO_synthetic_ablation.csv` | PR-AUC change when each feature block is removed, with a `beyond_noise` flag |
| `DEMO_synthetic_promiscuity.csv` | The four-way promiscuity interrogation across 5 seeds |
| `DEMO_synthetic_holdout_shortlist.csv` | Ranked pairs for a held-out gene, in-domain rows first, OOD rows appended and flagged |
| `_runtimes.json` | Wall-clock per notebook, written by the executor |

## Figures

`figures/` — all carry a `SYNTHETIC` watermark. The categorical palette is
validated for colour-vision-deficiency separation; see `../cookbook/_viz.py`
for the validator output and the rules followed.
