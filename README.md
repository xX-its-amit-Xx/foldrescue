# foldrescue

**Pair-level prediction of pharmacological chaperone rescue for destabilising
missense variants — with calibrated probabilities and an explicit
applicability domain.**

[![CI](https://github.com/xX-its-amit-Xx/foldrescue/actions/workflows/ci.yml/badge.svg)](https://github.com/xX-its-amit-Xx/foldrescue/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

> **Status: research software, demonstration results only.** The shipped
> notebooks and figures are executed on a **synthetic cohort**, because the
> curated benchmark this package is designed to consume
> ([`amenabench`](https://github.com/xX-its-amit-Xx/amenabench)) does not yet
> contain data. Every number in this repository describes the *software*, not
> biology. See [What this cannot do](#what-this-cannot-do).

---

## The question this package exists to answer

A **missense variant** changes one amino acid in a protein. Many such variants
do not destroy the protein's chemistry — they destabilise its *fold*. The
protein can still work, but it misfolds often enough that the cell's quality
control machinery (the endoplasmic-reticulum chaperone system) catches it and
destroys it before it reaches where it is needed. The enzyme is fine; it never
arrives. This is the mechanism behind a large share of lysosomal storage
diseases.

A **pharmacological chaperone** is a small molecule that binds the folded form
of the protein. Because it binds the folded state and not the unfolded one, it
shifts the folding equilibrium — by Le Chatelier's principle, removing folded
protein into a bound complex pulls more protein into the folded state. Enough
of it escapes quality control, traffics correctly, and does its job. The drug
does not repair the mutation. It changes the odds of folding.

A variant that responds this way is called **amenable** to that chaperone.

### Why existing tools do not answer it

Amenability is a property of a **(variant, compound) pair**. The two obvious
classes of tool each answer a different question:

| Tool | Question it answers | Question it does *not* answer |
|---|---|---|
| Stability predictors (FoldX, Rosetta, ThermoNet, ESM-based ΔΔG) | Does this variant destabilise the protein? | Whether *any* ligand helps |
| Docking / affinity prediction | Does this ligand bind this pocket? | Whether binding *rescues trafficking* |

Neither composes into the answer. A variant can be severely destabilised and
not amenable (misfolds before the pocket exists to bind). A compound can bind
tightly and rescue nothing (binds the already-folded population, shifting
nothing). Amenability depends on the *interaction* between a specific
perturbation and a specific ligand — which is why `foldrescue` models the pair
jointly rather than composing two single-sided predictors.

And because amenability datasets are small, heterogeneous and
assay-dependent, the second thing this package does is as important as the
first: **it reports when a query falls outside the distribution it was trained
on**, and refuses to rank those queries alongside supported ones.

---

## Install

```bash
# Tiers 1 and 3, calibration, the applicability domain and the CLI:
pip install -e .

# Add Tier 2 (frozen-encoder fusion; pulls in torch + transformers):
pip install -e ".[fusion]"

# Everything, including the cookbook and dev tooling:
pip install -e ".[all]"
```

RDKit is easier from conda on some platforms — see `environment.yml`.

## Use

```bash
foldrescue predict \
    --gene GLB1 \
    --variants variants.csv \
    --compounds compounds.smi \
    --out ranked.csv
```

Output columns:

| Column | Meaning |
|---|---|
| `variant`, `compound` | The pair |
| `p_amenable` | Raw model probability |
| `calibrated_p` | After temperature scaling or isotonic regression |
| `ci_low`, `ci_high` | Venn–Abers probability interval |
| `out_of_domain` | `True` if either domain axis failed |
| `nearest_training_neighbour` | Closest training compound, with its Tanimoto distance |
| `top_shap_features` | Signed top-k SHAP contributions for this row |
| `conformal_label_set` | Split-conformal labels not rejected; `0/1` means abstention |
| `ood_reason` | Which axis failed and by how much |

**By default the CLI will not rank out-of-domain pairs alongside in-domain
ones.** It writes the supported rows, reports how many were withheld and why,
and points you at `--include-ood`. This is deliberate friction: a ranked CSV
gets read from the top, and an extrapolation's `0.78` looks exactly like a
supported `0.78`. With `--include-ood`, flagged rows are appended *after* every
in-domain row and never co-sorted with them.

---

## Architecture: three tiers, so the comparison is honest

"We built a cross-attention transformer" is not a finding. "A cross-attention
transformer does not beat gradient-boosted trees at this sample size" is. All
three tiers share one interface so the harness cannot accidentally give one of
them more information than another.

### Tier 1 — interpretable baseline (`tier1_gbm`)

LightGBM over concatenated feature blocks:

- **stability** — predicted ΔΔG and its square (rescuability is non-monotonic
  in ΔΔG: a barely-destabilised protein does not need a chaperone, a
  catastrophically misfolded one cannot be saved by one), relative solvent
  accessibility, residue depth, conservation, secondary structure, and the
  *change* in hydropathy / side-chain volume / charge on substitution.
- **ligand** — ECFP4 Morgan count fingerprint plus interpretable RDKit
  physicochemical descriptors.
- **interaction** — a handful of mechanistically-motivated explicit products
  (ΔΔG × logP, pocket distance × TPSA, …), because a tree can only represent an
  interaction if both columns are reachable within one tree, and at n ≈ 10³
  they often are not.

On tabular data at this sample size **this is the model to beat, and it usually
is not beaten.** That is the point of having it.

### Tier 2 — representation fusion (`tier2_fusion`)

A frozen **ESM-2** embedding of the mutant sequence window and a frozen
**ChemBERTa** embedding of the compound, joined by a cross-attention block and
an MLP head.

- *Frozen by default.* ESM-2 has 10⁷–10⁸ parameters; a chaperone benchmark has
  ~10³ labelled pairs. Fine-tuning does not risk overfitting, it **guarantees**
  it — the encoder can memorise every training pair exactly.
  `unfreeze_encoders=True` exists because reviewers ask for it, and it logs a
  warning saying this.
- *Cross-attention, not concatenation.* Concatenating two vectors forces the
  first dense layer to learn every pair interaction as a linear map of the
  joint vector. Cross-attention lets the compound query the protein
  representation directly, which is a better inductive bias for "which part of
  this molecule matters for this residue?".
- *Variant-centred pooling.* The protein side pools a narrow span around the
  variant position, not the whole sequence. Whole-sequence mean pooling makes
  every variant of a protein collapse to nearly the same vector — an easy
  failure mode to ship by accident.

### Tier 3 — structure-aware (`tier3_structure`)

Tier 1 **plus** pocket-occupancy features: Cα-to-pocket distance (with
exponential decay terms, since contact effects fall off sharply), pocket volume
change on mutation, and whether the residue lies in a loop contacting the
ligand.

Deliberately **architecturally identical to Tier 1**, so any difference in
numbers is attributable to the structural information and nothing else.
Comparing a GBM to a GBM-with-extra-columns is a clean experiment; comparing a
GBM to a different architecture with extra columns is not.

---

## Calibration and honesty — the point, not a footnote

### Discrimination ≠ calibration

A ranking model tells you *which* pair is most promising. A **calibrated** model
tells you *how likely* a pair is to work — and only the second supports the
decision this package exists to inform, which is how to spend a finite assay
budget. The two properties are mathematically independent: you can have either
without the other, and boosted trees trained on imbalanced data are routinely
excellent at ranking and badly overconfident about probabilities.

`foldrescue` fits calibration on a **dedicated held-out calibration split**,
never on test. Every split in this package is three-way (train / calib / test)
for exactly this reason: a temperature fitted on the test set is a fit
statistic, not a measurement.

- **Temperature scaling** — one scalar divides the logit. Being monotone with a
  single parameter, it **provably cannot change the ranking** (AUC is identical
  before and after — this is asserted in the test suite) and cannot overfit a
  small calibration split. The right default.
- **Isotonic regression** — a free monotone step function. More expressive, so
  it can fix non-monotone miscalibration, at the cost of overfitting a few
  hundred calibration points and producing ties.

Reported before and after: **Brier score** (a proper scoring rule, so it
captures discrimination and calibration together) and **expected calibration
error** with *equal-count* bins by default, because equal-width ECE can hide
most of the error in one crowded bin.

### Applicability domain

A model returns a number for any input. Ask a model trained on iminosugars
about a macrocyclic peptide and it answers `0.73` with no hesitation. That
number is an extrapolation wearing a prediction's clothes.

The applicability domain is the region where a model's error rate has actually
been *estimated*. It is a long-settled requirement in QSAR — principle 3 of the
OECD's five validation principles for (Q)SAR models, adopted in 2007 precisely
because regulators kept receiving confident predictions about chemicals unlike
anything in the training set.

Two axes, checked independently:

- **Compound axis** — Tanimoto distance to the nearest *training* compound.
  Tanimoto (Jaccard) on binary fingerprints is shared substructures over total
  substructures. The conventional "similar enough" line is T ≥ 0.4 on ECFP4, so
  the default threshold is a distance of **0.60**. This is a convention, not a
  law of nature, and it is a parameter.
- **Protein axis** — sequence separation to the nearest trained position
  *within the same gene* (default **25 residues**, roughly a secondary-structure
  element plus flanks). A gene absent from training is out of domain by
  construction — which is the honest answer for a prospective query on a new
  gene.

A query is out of domain if **either** axis fails.

### Uncertainty: two routes, two different guarantees

- **Venn–Abers probability intervals** (`ci_low`, `ci_high`) are the default.
  They are *valid*: one endpoint is a perfectly calibrated probability, so the
  interval is guaranteed to bracket a calibrated estimate, and its width reads
  out how much calibration data constrains that region.
- **Split-conformal prediction** gives finite-sample **marginal** coverage
  under exchangeability alone, with no assumption that the model is any good.
  The catch is the word *marginal*: coverage holds on average over the
  population, **not** conditionally for your particular query. Its useful
  output for a binary label is the **label set** — `{0}`, `{1}`, or `{0,1}`
  (an explicit abstention).

> **A note on an interval we removed.** An earlier version computed
> `ci_low`/`ci_high` by split-conformal regression on `|y − p|`. With binary
> labels the calibration residuals are large, the quantile lands near 0.9, and
> **every interval collapsed to [0, 1]** — formally valid, completely
> uninformative. It is recorded here because that failure is easy to ship and
> hard to notice: the column is populated, the method is citable, and the
> output means nothing.

Conformal intervals and the applicability domain are **complements, not
substitutes**: the interval quantifies noise *assuming* exchangeability, while
the domain flag tells you whether exchangeability is plausible at all.

### Did the model learn biology, or just which compounds are promiscuous?

This is the question that decides whether the package is useful, and neither
SHAP nor permutation importance answers it. A model can have beautiful
attribution plots dominated by ligand features while doing nothing but
recalling per-compound base rates.

`foldrescue.attribution.interrogate_promiscuity` runs four nested comparisons:

1. **Compound base-rate baseline** — predict each compound's training positive
   rate, ignoring the variant entirely. Any claim of pair-level modelling must
   beat this.
2. **Variant base-rate baseline** — the symmetric check.
3. **Within-variant compound shuffle** — permute compound identity among pairs
   sharing a variant. Compound marginals are preserved *exactly*, so a
   promiscuity-only model is unaffected, while genuine pair-specific signal is
   destroyed. The drop isolates pair specificity.
4. **Recovered vs. generative pair-signal share** — on the synthetic cohort the
   true mixture weights are known, so the diagnostic itself can be validated.
   Real data never offers this check, which is precisely why it is worth having
   on synthetic data.

The report returns a sentence, not a number to squint at — e.g. *"MOSTLY
PROMISCUITY: just 12% of the lift is pair-specific. The model is close to a
per-compound base-rate predictor dressed up as a pair model."*

---

## Resource guard

Every allocation-heavy path passes through `check_resources()` **before**
allocating:

1. Read **available** (not total) RAM and CPU count via `psutil`.
2. Estimate the dense footprint: `rows × cols × itemsize × working_copies`.
3. If it exceeds **50%** of available RAM, mitigate in a fixed order:
   **downcast** float64 → float32 → **chunk** along rows → **cap `n_jobs` at
   `nproc − 1`** (every worker is another copy of the working set, and leaving
   one core free keeps the box responsive).
4. Otherwise **refuse**, naming the numbers and the specific knobs that would
   make the job fit.

A refusal is a *success* of this function; the alternative is an OOM kill
thirty minutes into a run. Why 50% and not 90%? The estimate is a lower bound —
Python objects, BLAS scratch and the interpreter all live outside it.

```bash
foldrescue resources --rows 5000000 --cols 100000 --no-chunking
# RESOURCE REFUSAL
# foldrescue refuses the requested allocation: it needs 3.64 TiB but the
# budget is 7.48 GiB (14.96 GiB available x memory_fraction). ...
```

Embedding extraction **streams and checkpoints**: batches are flushed to disk
immediately, a manifest records completed ids plus a fingerprint of the
extraction configuration, and a rerun resumes. The fingerprint matters — resuming
a cache built with a different model or pooling strategy would silently mix two
incompatible representations into one matrix.

The contract is documented in `foldrescue/utils.py`, [`CLAUDE.md`](CLAUDE.md)
and [`AGENTS.md`](AGENTS.md).

---

## Data layer

`foldrescue` consumes a benchmark through the `BenchmarkAdapter` protocol.

- **`AmenabenchAdapter`** reads curated CSVs in the schema documented in
  [`docs/data_contract.md`](docs/data_contract.md). As of this writing the
  `amenabench` repository contains only a `LICENSE`, so this adapter raises an
  error naming the paths it looked for. `foldrescue` therefore **defines** this
  contract rather than mirroring one.
- **`SyntheticCohortAdapter`** generates a seeded cohort with *known*
  generative mixture weights. It is a test fixture at benchmark scale, not a
  dataset, and it exists so the calibration and attribution machinery can be
  validated against an answer we already know.

Four named splits, all grouped where grouping matters: `random`,
`gene_holdout`, `compound_holdout`, `scaffold`. The honest number for a
prospective question comes from the holdout that matches it — "will this work
on a gene I have never trained on?" is answered only by `gene_holdout`, and the
test suite asserts that no gene straddles that split.

---

## Cookbook

| Notebook | What it does |
|---|---|
| `01_train_all_tiers.ipynb` | All three tiers across every split; PR-AUC / MCC / Brier with error bars over 5 seeds |
| `02_calibration_study.ipynb` | Reliability diagrams before/after; what uncalibrated confidence would have cost |
| `03_glb1_prospective.ipynb` | Train excluding one gene, produce a ranked shortlist, mark what is in-domain |
| `04_ablation.ipynb` | Remove the stability, ligand, then structural block; report what each contributes |

Executed outputs, figures under `results/figures/`, and versions plus runtimes
in [`RUN_ENVIRONMENT.md`](RUN_ENVIRONMENT.md).

**All cookbook results are computed on the synthetic cohort** and are watermarked
accordingly. They demonstrate that the pipeline runs and that the diagnostics
behave as designed. They are not findings about chaperone biology.

---

## What this cannot do

Read this section before using any output for anything.

**Nothing here is clinical decision support.** It must not be used to select,
withhold, dose or sequence therapy for any patient. It is not a medical device,
has not been validated for clinical use, and has not been reviewed by any
regulatory body.

**In vitro amenability is not a guarantee of in vivo clinical response.** A
variant can be amenable in a cell assay and the patient still not benefit.
Rescued enzyme must reach the right tissue at the right time in sufficient
quantity; the chaperone must be orally available, reach the target tissue,
cross the relevant barriers, and not inhibit the very enzyme it stabilises at
therapeutic concentrations — many pharmacological chaperones are active-site
binders and are therefore inhibitors at higher occupancy, which makes the
therapeutic window narrow and compound-specific. Residual enzyme activity
thresholds differ by tissue and by disease. None of this is modelled here.

**Predictions are hypotheses for assay prioritisation only.** The intended use
is deciding which of many (variant, compound) pairs to test *first*. The output
is a reordering of an experimental queue, not evidence about any pair.

**The shipped numbers are from synthetic data.** The demonstration cohort is
generated by `foldrescue.data.SyntheticCohortAdapter`. Its labels encode a
mixture *we* chose. Metrics on it measure whether the software works, and carry
no biological information whatsoever. Until a curated benchmark is supplied via
`--cohort <path>`, **every probability this package emits is a software
demonstration.**

**A prediction inside the applicability domain is still not a guarantee.** The
in-domain flag says the query resembles data on which error was *measured*. The
model can still be wrong — just wrong at approximately the rate the benchmark
reported, rather than at an unknown rate.

**Calibration is split- and population-specific.** A temperature fitted on one
calibration split describes that distribution. It does not transfer to a
different gene panel, a different assay family, or a different laboratory.

**The structural features are only as good as the structure.** Pocket distance
and volume change assume a correct structure, correct isoform numbering, and a
correctly identified binding pocket. A predicted structure with a mismodelled
loop, or an off-by-one isoform offset, produces confident features about the
wrong residue. `foldrescue` warns when a supplied sequence disagrees with the
stated wild-type residue, but it cannot detect a wrong pocket.

**Absence of evidence is not negative evidence.** A low `p_amenable` for a pair
is not grounds to conclude the pair does not work, particularly for
under-represented genes and chemotypes. The model has seen very little, and
most of what it has seen is positive-reporting-biased: the literature publishes
rescues more readily than failures.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check foldrescue tests
```

Required behaviours covered by tests: calibration reduces ECE on a synthetic
miscalibrated case (and recovers the planted temperature); the
applicability-domain flag fires on a deliberately out-of-distribution compound
*and does not fire on a close analogue*; the resource guard refuses correctly
and its refusal is actionable; the CLI withholds out-of-domain pairs.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).
