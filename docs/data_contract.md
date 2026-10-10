# The benchmark data contract

`foldrescue` consumes a curated benchmark through
`foldrescue.data.BenchmarkAdapter`. This document is the normative schema.

`foldrescue` **defines** this contract rather than inheriting it: the
`amenabench` repository it was designed to consume currently contains only a
`LICENSE`, so there was no contract to mirror. If you are building that
benchmark, this file is the target.

## Files

A benchmark root contains:

```
<root>/
  variants.csv     one row per variant, first column is the variant_id index
  compounds.csv    one row per compound
  pairs.csv        one row per (variant, compound) observation
  splits.json      optional but strongly preferred (see below)
```

### `variants.csv`

| Column | Type | Meaning |
|---|---|---|
| *(index)* | str | `variant_id`, unique |
| `gene` | str | HGNC symbol, e.g. `GLB1` |
| `uniprot` | str | Accession for the canonical isoform, e.g. `P16278` |
| `position` | int | **1-based** residue position in that isoform |
| `wt_aa`, `mut_aa` | str | One-letter amino acids |
| `ddg_pred` | float | Predicted folding ΔΔG, kcal/mol, **positive = destabilising** |
| `rsa` | float | Relative solvent accessibility of the WT residue, 0–1 |
| `sec_struct` | str | `H` (helix), `E` (strand), `L` (loop) |
| `conservation` | float | Per-position conservation, 0–1 |
| `residue_depth` | float | Distance from the protein surface, Å |
| `dist_to_pocket` | float | Cα-to-pocket-centroid distance, Å — *Tier 3* |
| `pocket_vol_delta` | float | Pocket volume change on mutation, Å³ — *Tier 3* |
| `contacts_ligand` | int | 0/1, residue in a loop contacting the ligand — *Tier 3* |

**Isoform numbering is the most common source of silent error.** `position`
must index the same isoform the `uniprot` accession names. An off-by-one offset
produces confident features about the wrong residue. `foldrescue` warns when a
supplied sequence disagrees with `wt_aa`, which catches many but not all cases.

**Sign convention for `ddg_pred` matters.** Positive = destabilising. Half the
tools in this field use the opposite sign; converting wrong inverts the single
most informative feature.

### `compounds.csv`

| Column | Type | Meaning |
|---|---|---|
| `compound_id` | str | Unique; also used as the frame index |
| `smiles` | str | Must parse under RDKit — unparseable rows are **rejected**, not skipped |
| `mechanism` | str | Free text, e.g. `active-site iminosugar` |

Stereochemistry should be specified where it is known. Iminosugar chaperones are
frequently stereochemistry-dependent, and a flat SMILES silently merges
diastereomers with different activity into one fingerprint.

### `pairs.csv`

| Column | Type | Meaning |
|---|---|---|
| `variant_id` | str | Foreign key into `variants.csv` |
| `compound_id` | str | Foreign key into `compounds.csv` |
| `label` | int | `1` = rescued, `0` = not rescued, **in the source assay** |
| `assay` | str | Assay family; splits may group on this |
| `source` | str | DOI or accession — required, for provenance |

Dangling foreign keys and labels outside {0,1} are rejected on load.

#### On what `label` means

`label` is assay-relative, and this is the benchmark's hardest problem rather
than a detail to normalise away:

- **Thresholds differ.** "Rescue" might mean ≥1.5-fold activity increase in one
  paper and ≥20% of wild-type activity in another. Record the assay in `assay`
  and keep the threshold documented per source.
- **Readouts differ.** Enzyme activity, thermal stabilisation and trafficking
  (immunofluorescence / glycosylation state) measure different things and
  disagree. A thermal shift is not a rescue.
- **Publication bias is positive.** Rescues are published far more readily than
  failures, so a benchmark assembled from the literature over-represents
  positives — which is why a low predicted probability is weak negative
  evidence.
- **Concentration matters.** Many chaperones are active-site binders, hence
  inhibitors at higher occupancy. A pair labelled positive at 10 µM may be
  negative at 100 µM. Record it.

Where the same pair is measured more than once, keep both rows with distinct
`source` values rather than silently resolving the conflict; disagreement
between assays is signal about label noise.

### `splits.json`

```json
{
  "gene_holdout": {
    "train": [0, 1, 2],
    "calib": [3, 4],
    "test":  [5, 6, 7]
  }
}
```

Integer **row positions** into `pairs.csv`. Every split must be three-way:
`train`, `calib`, `test`. The calibration partition is not optional — see
`CLAUDE.md` §3.

Predefined splits are strongly preferred. Locally derived splits are not
comparable across papers, and `foldrescue` logs a warning when it has to
derive its own.

## Splits `foldrescue` derives when none are supplied

| Name | Grouped by | Question it answers |
|---|---|---|
| `random` | nothing | Interpolation within the known space (optimistic) |
| `gene_holdout` | `gene` | "A protein I have never trained on" |
| `compound_holdout` | `compound_id` | "A compound I have never trained on" |
| `scaffold` | Bemis–Murcko scaffold | "A chemotype I have never trained on" |

Use the holdout that matches your question. A random split over a congeneric
series flatters a model badly, because the test set is full of near-twins of
the training set.

## Minimal adapter for a different source

```python
from dataclasses import dataclass
from foldrescue.data import Cohort, build_splits

@dataclass
class MyAdapter:
    name: str = "my-benchmark"

    def load(self) -> Cohort:
        cohort = Cohort(
            variants=...,      # indexed by variant_id
            compounds=...,     # indexed by compound_id
            pairs=...,
            synthetic=False,
            provenance="MyAdapter: curated from <source>, accessed <date>",
        )                      # validates on construction
        cohort.splits = build_splits(cohort, seed=0)
        return cohort
```

Set `synthetic=False` **only** for curated measurements. The flag drives every
warning banner and watermark in the package.
