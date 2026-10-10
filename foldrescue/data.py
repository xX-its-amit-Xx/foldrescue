"""Data layer: the benchmark adapter seam, and a synthetic cohort to test it.

WHY THERE IS A SEAM HERE
========================

foldrescue predicts over *(variant, compound)* pairs. It does not own the
curation of those pairs -- that is a benchmark's job, and curating amenability
labels from the literature is slow, contested, assay-dependent work.

So the data layer is a protocol, ``BenchmarkAdapter``, with two
implementations:

``SyntheticCohortAdapter``
    A seeded generator with *known* ground truth. Not a stand-in for real data
    and not presented as one: it exists so the pipeline, the calibration
    machinery, and above all the attribution analysis can be tested against an
    answer we already know. Everything it produces is labelled SYNTHETIC.

``AmenabenchAdapter``
    Reads a curated benchmark laid out in the schema documented below. As of
    this writing the ``amenabench`` repository contains only a LICENSE, so this
    adapter raises a clear error naming what it looked for. It is wired, not
    guessed-at: the schema here *is* the contract, and foldrescue is its
    reference implementation.

THE CONFOUND THE SYNTHETIC COHORT DELIBERATELY CONTAINS
=======================================================

The central scientific risk in pair prediction is trivial: a model can score
well without learning any structure-function relationship at all, simply by
learning *which compounds rescue lots of things*. If 30% of your positives
come from three promiscuous compounds, "predict the compound's base rate" is a
strong baseline that looks like insight.

``SyntheticCohortAdapter`` therefore generates labels from an explicit mixture:

    logit(p) = w_pair * pair_interaction_term
             + w_promisc * compound_promiscuity_term
             + w_stability * variant_stability_term
             + noise

with the weights stored on the returned ``Cohort`` as
``ground_truth_weights``. That turns "did the model learn real signal or just
compound identity?" from a rhetorical question into a measurable one: the
attribution analysis in ``attribution.py`` recovers an estimated mixture and
can be checked against the generative truth. On real data you never get this
check -- which is exactly why it is worth having on synthetic data.

SCHEMA (the contract a real benchmark must satisfy)
===================================================

``variants.csv``   one row per variant
    gene               str    HGNC symbol, e.g. GLB1
    uniprot            str    accession, e.g. P16278
    position           int    1-based residue position in the canonical isoform
    wt_aa, mut_aa      str    one-letter amino acids
    ddg_pred           float  predicted folding dG change, kcal/mol (+ = destabilising)
    rsa                float  relative solvent accessibility of the WT residue, 0-1
    sec_struct         str    H | E | L  (helix / strand / loop)
    conservation       float  per-position conservation, 0-1
    residue_depth      float  distance from the protein surface, angstrom
    dist_to_pocket     float  CA-to-pocket-centroid distance, angstrom  (Tier 3)
    pocket_vol_delta   float  pocket volume change on mutation, A^3       (Tier 3)
    contacts_ligand    int    0/1, variant in a loop contacting the ligand (Tier 3)

``compounds.csv``  one row per compound
    compound_id        str
    smiles             str    parsed with RDKit; unparseable rows are rejected
    mechanism          str    free text, e.g. "active-site iminosugar"

``pairs.csv``      one row per (variant, compound) observation
    variant_id         str    foreign key into variants.csv index
    compound_id        str    foreign key into compounds.csv
    label              int    1 = rescued, 0 = not rescued, in the source assay
    assay              str    assay family; splits may group on this
    source             str    DOI or accession, for provenance

Splits are named and explicit (``random``, ``gene_holdout``,
``compound_holdout``, ``scaffold``) because the honest number for a
prospective question is the one from the holdout that matches it: asking "will
this work on a gene I have never trained on?" is only answered by
``gene_holdout``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .utils import get_logger

__all__ = [
    "Cohort",
    "BenchmarkAdapter",
    "SyntheticCohortAdapter",
    "AmenabenchAdapter",
    "SplitName",
    "load_cohort",
    "AMINO_ACIDS",
]

_log = get_logger(__name__)

SplitName = Literal["random", "gene_holdout", "compound_holdout", "scaffold"]

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Required columns, mirrored from the module docstring. Kept as data so the
# adapters validate against exactly what the docs promise.
VARIANT_COLUMNS = (
    "gene",
    "uniprot",
    "position",
    "wt_aa",
    "mut_aa",
    "ddg_pred",
    "rsa",
    "sec_struct",
    "conservation",
    "residue_depth",
    "dist_to_pocket",
    "pocket_vol_delta",
    "contacts_ligand",
)
COMPOUND_COLUMNS = ("compound_id", "smiles", "mechanism")
PAIR_COLUMNS = ("variant_id", "compound_id", "label", "assay", "source")


@dataclass
class Cohort:
    """A benchmark instance: variants, compounds, observed pairs, and splits.

    Attributes
    ----------
    variants:
        Indexed by ``variant_id``. Columns per ``VARIANT_COLUMNS``.
    compounds:
        Indexed by ``compound_id``. Columns per ``COMPOUND_COLUMNS``.
    pairs:
        One row per observation. Columns per ``PAIR_COLUMNS``.
    splits:
        ``{split_name: {"train"|"calib"|"test": np.ndarray of pair row indices}}``.
        Three-way rather than two-way because calibration *must* be fitted on
        data the model never saw and never validated against -- a temperature
        fitted on the test set is not a measurement, it is a fit.
    synthetic:
        True when the cohort was generated rather than curated. Propagated into
        every figure title, every CSV header, and the CLI banner.
    ground_truth_weights:
        For synthetic cohorts, the generative mixture weights. ``None`` for real
        data, where no such thing is knowable.
    oracle_p:
        For synthetic cohorts, the true generative probability per pair. Held
        here rather than as a column on ``pairs`` so it can never be swept into
        a feature matrix by accident. Its AUC is the Bayes ceiling: the best any
        model could do against this much label noise. Reporting a model's number
        without it invites the reader to mistake an irreducible-noise floor for
        a modelling failure.
    provenance:
        Free-form record of where the data came from, written into
        ``RUN_ENVIRONMENT.md`` and the output CSV header.
    """

    variants: pd.DataFrame
    compounds: pd.DataFrame
    pairs: pd.DataFrame
    splits: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    synthetic: bool = False
    ground_truth_weights: dict[str, float] | None = None
    oracle_p: np.ndarray | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail loudly on a malformed cohort, naming the exact problem.

        Called on construction. A benchmark that silently loses its foreign
        keys produces a model that silently trains on garbage.
        """
        missing_v = set(VARIANT_COLUMNS) - set(self.variants.columns)
        if missing_v:
            raise ValueError(
                f"variants frame is missing required columns: {sorted(missing_v)}. "
                f"See foldrescue.data module docstring for the schema."
            )
        missing_c = set(COMPOUND_COLUMNS) - set(self.compounds.columns)
        if missing_c:
            raise ValueError(
                f"compounds frame is missing required columns: {sorted(missing_c)}."
            )
        missing_p = set(PAIR_COLUMNS) - set(self.pairs.columns)
        if missing_p:
            raise ValueError(f"pairs frame is missing required columns: {sorted(missing_p)}.")

        unknown_v = set(self.pairs["variant_id"]) - set(self.variants.index)
        if unknown_v:
            raise ValueError(
                f"{len(unknown_v)} pair rows reference variant_ids absent from the "
                f"variants frame, e.g. {sorted(unknown_v)[:3]}."
            )
        unknown_c = set(self.pairs["compound_id"]) - set(self.compounds.index)
        if unknown_c:
            raise ValueError(
                f"{len(unknown_c)} pair rows reference compound_ids absent from the "
                f"compounds frame, e.g. {sorted(unknown_c)[:3]}."
            )
        bad_labels = set(np.unique(self.pairs["label"])) - {0, 1}
        if bad_labels:
            raise ValueError(f"labels must be 0/1, found {sorted(bad_labels)}.")

    # -- convenience ------------------------------------------------------

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def prevalence(self) -> float:
        """Fraction of positive pairs. Printed next to every PR-AUC, because
        PR-AUC is only interpretable relative to the base rate."""
        return float(self.pairs["label"].mean()) if len(self.pairs) else float("nan")

    def joined(self) -> pd.DataFrame:
        """Pairs with variant and compound attributes merged in."""
        out = self.pairs.join(self.variants, on="variant_id")
        return out.join(self.compounds, on="compound_id", rsuffix="_cmp")

    def split(self, name: str) -> dict[str, np.ndarray]:
        if name not in self.splits:
            raise KeyError(
                f"unknown split {name!r}; available: {sorted(self.splits)}. "
                f"Pick the split that matches your question -- gene_holdout for "
                f"'a gene I have never trained on'."
            )
        return self.splits[name]

    def label_banner(self) -> str:
        """The line that goes on every artefact derived from this cohort."""
        if self.synthetic:
            return (
                "SYNTHETIC COHORT -- generated data, not experimental measurement. "
                "Metrics below describe the pipeline, NOT biology."
            )
        return f"Curated cohort: {self.provenance}"


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """The seam. Anything satisfying this can feed foldrescue."""

    name: str

    def load(self) -> Cohort:
        """Return a validated ``Cohort``, or raise with an actionable message."""
        ...


# ---------------------------------------------------------------------------
# Synthetic cohort
# ---------------------------------------------------------------------------

# A small set of real iminosugar / aminocyclitol-like scaffolds plus some
# decoys. These are real SMILES because the *chemistry featurisation* must be
# exercised for real -- ECFP on a fake SMILES tests nothing. The LABELS
# attached to them are synthetic and carry no experimental meaning.
_SCAFFOLD_SMILES: tuple[tuple[str, str, str], ...] = (
    ("CMP_IMINO_01", "OC[C@H]1NC[C@H](O)[C@@H](O)[C@@H]1O", "iminosugar (deoxynojirimycin-like)"),
    ("CMP_IMINO_02", "OC[C@H]1N[C@H](CO)[C@@H](O)[C@@H]1O", "iminosugar, truncated"),
    ("CMP_IMINO_03", "CCCCN1C[C@H](O)[C@@H](O)[C@H](O)[C@H]1CO", "N-alkyl iminosugar"),
    ("CMP_AMINO_01", "N[C@H]1[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O", "aminocyclitol"),
    ("CMP_AZA_01", "OC[C@@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "pyranose control"),
    ("CMP_ARYL_01", "O=C(Nc1ccc(F)cc1)c1ccc(O)cc1", "aryl amide, allosteric-type"),
    ("CMP_ARYL_02", "O=C(Nc1ccccc1)c1ccc(Cl)cc1", "aryl amide analogue"),
    ("CMP_ARYL_03", "COc1ccc(C(=O)Nc2ccccc2)cc1", "aryl amide, methoxy"),
    ("CMP_HETERO_01", "c1ccc2[nH]c(C(=O)NCCN)cc2c1", "indole carboxamide"),
    ("CMP_HETERO_02", "c1ccc2nc(N3CCOCC3)ccc2c1", "quinoline morpholine"),
    ("CMP_DECOY_01", "CCCCCCCCCCCCCCCC(=O)O", "long-chain fatty acid (decoy)"),
    ("CMP_DECOY_02", "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1", "terphenyl (decoy, insoluble)"),
)

# Placeholder gene panel. UniProt accessions are real so that anyone wiring a
# real structure source in later finds the identifiers already correct; the
# per-variant numbers attached to them are generated.
def _compound_descriptor_table(compounds: pd.DataFrame) -> pd.DataFrame:
    """Real RDKit descriptors per compound, used by the synthetic generator.

    Deliberately computed from the actual SMILES rather than drawn at random:
    the generative interaction term must be a function of quantities the
    feature matrix also contains, or it is unlearnable (see ``_make_pairs``).
    """
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import Descriptors  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    wanted = ("MolWt", "MolLogP", "TPSA", "NumRotatableBonds", "FractionCSP3")
    rows = {}
    for cid, smi in compounds["smiles"].items():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"synthetic cohort contains an unparseable SMILES for {cid}: {smi!r}")
        rows[cid] = {w: float(getattr(Descriptors, w)(mol)) for w in wanted}
    return pd.DataFrame.from_dict(rows, orient="index")


_GENE_PANEL: tuple[tuple[str, str, int], ...] = (
    ("GLB1", "P16278", 677),
    ("GBA1", "P04062", 536),
    ("GALC", "P54803", 685),
    ("HEXA", "P06865", 529),
    ("IDUA", "P35475", 653),
    ("NAGLU", "P54802", 743),
    ("FUCA1", "P04066", 466),
    ("MAN2B1", "O00754", 1011),
)


@dataclass
class SyntheticCohortAdapter:
    """Seeded synthetic cohort with known generative structure.

    This is a *test fixture at benchmark scale*, not a dataset. Its reason for
    existing is the ``ground_truth_weights`` it reports: with them, the
    promiscuity analysis becomes falsifiable.

    Parameters
    ----------
    seed:
        Generator seed. All randomness derives from it.
    n_variants_per_gene:
        Variants sampled per gene in the panel.
    w_pair, w_promisc, w_stability:
        Generative mixture weights on the logit scale. The defaults encode a
        deliberately awkward situation: compound promiscuity (1.1) is a
        *stronger* single effect than the genuine pair interaction (0.9). A
        model that scores well here may well have learned the confound, and the
        attribution analysis is supposed to catch that.
    noise_sd:
        Logit-scale noise, i.e. irreducible error. Caps achievable performance,
        which is realistic and keeps the notebooks from reporting a suspicious
        0.99.
    """

    seed: int = 0
    n_variants_per_gene: int = 22
    w_pair: float = 0.9
    w_promisc: float = 1.1
    w_stability: float = 0.6
    noise_sd: float = 0.7
    name: str = "synthetic"

    def load(self) -> Cohort:
        rng = np.random.default_rng(self.seed)

        variants = self._make_variants(rng)
        compounds = self._make_compounds()
        pairs, weights, oracle_p = self._make_pairs(rng, variants, compounds)

        cohort = Cohort(
            variants=variants,
            compounds=compounds,
            pairs=pairs,
            synthetic=True,
            ground_truth_weights=weights,
            oracle_p=oracle_p,
            provenance=(
                f"foldrescue.data.SyntheticCohortAdapter(seed={self.seed}, "
                f"w_pair={self.w_pair}, w_promisc={self.w_promisc}, "
                f"w_stability={self.w_stability}, noise_sd={self.noise_sd})"
            ),
        )
        cohort.splits = build_splits(cohort, seed=self.seed)
        _log.info(
            "synthetic cohort: %d variants x %d compounds -> %d pairs, prevalence %.3f",
            len(variants),
            len(compounds),
            len(pairs),
            cohort.prevalence,
        )
        return cohort

    # -- generation pieces ------------------------------------------------

    def _make_variants(self, rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for gene, uniprot, length in _GENE_PANEL:
            # Each gene gets a "pocket" at a fixed fraction of its length, so
            # dist_to_pocket is internally consistent per gene.
            pocket_centre = int(length * 0.42)
            for _ in range(self.n_variants_per_gene):
                position = int(rng.integers(1, length + 1))
                wt, mut = rng.choice(list(AMINO_ACIDS), size=2, replace=False)
                # Buried residues: low RSA, high depth, and (realistically) a
                # larger destabilisation on mutation.
                rsa = float(np.clip(rng.beta(1.6, 3.2), 0.0, 1.0))
                depth = float(np.clip(12.0 * (1.0 - rsa) + rng.normal(0, 1.4), 0.3, 20.0))
                ddg = float(np.clip(rng.gamma(2.1, 0.78) * (1.35 - rsa), -0.6, 7.5))
                # Sequence distance to the pocket, mapped to a plausible
                # spatial distance with noise -- adjacent in sequence does not
                # mean adjacent in space, and the noise encodes that.
                seq_gap = abs(position - pocket_centre)
                dist = float(np.clip(2.6 + 0.055 * seq_gap + rng.normal(0, 3.1), 2.0, 48.0))
                rows.append(
                    {
                        "gene": gene,
                        "uniprot": uniprot,
                        "position": position,
                        "wt_aa": str(wt),
                        "mut_aa": str(mut),
                        "ddg_pred": round(ddg, 3),
                        "rsa": round(rsa, 3),
                        "sec_struct": str(rng.choice(["H", "E", "L"], p=[0.36, 0.26, 0.38])),
                        "conservation": round(float(np.clip(rng.beta(3.0, 1.7), 0, 1)), 3),
                        "residue_depth": round(depth, 2),
                        "dist_to_pocket": round(dist, 2),
                        # Mutations near the pocket perturb its volume more.
                        "pocket_vol_delta": round(
                            float(rng.normal(0, 26.0) * np.exp(-dist / 11.0)), 2
                        ),
                        "contacts_ligand": int(dist < 8.0 and rng.random() < 0.62),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.index = pd.Index(
            [f"{r.gene}_{r.wt_aa}{r.position}{r.mut_aa}_{i}" for i, r in enumerate(frame.itertuples())],
            name="variant_id",
        )
        return frame

    def _make_compounds(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [{"compound_id": c, "smiles": s, "mechanism": m} for c, s, m in _SCAFFOLD_SMILES]
        )
        frame.index = pd.Index(frame["compound_id"], name="compound_id")
        return frame

    def _make_pairs(
        self, rng: np.random.Generator, variants: pd.DataFrame, compounds: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, float], np.ndarray]:
        """Generate labels from the documented three-term mixture.

        CRITICAL DESIGN POINT -- the pair-interaction term is built from
        *observable* quantities (variant structural properties crossed against
        real RDKit descriptors of the compound), not from hidden latent
        vectors.

        An earlier version of this generator used random per-variant and
        per-compound latents. That was wrong, and wrong in an instructive way:
        the resulting pair signal was not a function of anything in the feature
        matrix, so it was *mathematically unlearnable*. Every model sat at
        chance on the term that carried the most generative weight, and the
        "did the model find real pair signal?" analysis was unanswerable by
        construction -- there was no recoverable signal to find.

        Tying the interaction to observables makes the central question of this
        package well-posed: the pair term is learnable in principle, the
        promiscuity term is learnable by memorising compound identity, and the
        attribution analysis can therefore distinguish a model that found the
        former from one that only exploited the latter.
        """
        # Per-compound promiscuity: a few compounds rescue many things. This is
        # the confound, and it is drawn deliberately skewed.
        promisc = pd.Series(
            rng.beta(1.3, 3.4, size=len(compounds)), index=compounds.index, name="promiscuity"
        )

        # Real physicochemical descriptors, z-scored. These are observable to
        # the model through the ligand feature block, so an interaction built
        # on them is recoverable.
        cmp_desc = _compound_descriptor_table(compounds)
        cmp_z = (cmp_desc - cmp_desc.mean()) / cmp_desc.std(ddof=0).replace(0.0, 1.0)

        # Variant-side observables, z-scored the same way.
        var_cols = ["ddg_pred", "rsa", "dist_to_pocket", "conservation"]
        var_z = (variants[var_cols] - variants[var_cols].mean()) / variants[var_cols].std(
            ddof=0
        ).replace(0.0, 1.0)

        # Fixed mechanistic cross-terms with signs chosen to be chemically
        # sensible: a greasier, heavier ligand helps a more destabilised fold;
        # a polar ligand does worse at a buried site; pocket-distal variants
        # benefit less from a rigid ligand.
        cross: tuple[tuple[str, str, float], ...] = (
            ("ddg_pred", "MolLogP", +0.85),
            ("ddg_pred", "MolWt", +0.55),
            ("rsa", "TPSA", -0.70),
            ("dist_to_pocket", "NumRotatableBonds", -0.60),
            ("conservation", "FractionCSP3", +0.45),
        )
        cmp_pos = {c: i for i, c in enumerate(compounds.index)}
        var_pos = {v: i for i, v in enumerate(variants.index)}
        var_zv = var_z.to_numpy(float)
        cmp_zv = cmp_z.to_numpy(float)
        var_ix = {c: i for i, c in enumerate(var_cols)}
        cmp_ix = {c: i for i, c in enumerate(cmp_z.columns)}
        scale = float(np.sqrt(sum(w * w for _, _, w in cross)))

        rows: list[dict] = []
        oracle: list[float] = []
        # Not a full cross product: real benchmarks are sparse and ragged,
        # with each compound tested against a biased subset of variants.
        for variant_id, v in variants.iterrows():
            n_tested = int(rng.integers(3, len(compounds) + 1))
            tested = rng.choice(compounds.index.to_numpy(), size=n_tested, replace=False)
            for compound_id in tested:
                i, j = var_pos[variant_id], cmp_pos[compound_id]
                pair_term = (
                    sum(
                        w * var_zv[i, var_ix[pv]] * cmp_zv[j, cmp_ix[cd]]
                        for pv, cd, w in cross
                        if cd in cmp_ix
                    )
                    / scale
                )
                promisc_term = float(promisc[compound_id]) * 2.0 - 1.0
                # Chaperone rescue is most plausible for *mildly* destabilised,
                # buried, folding-competent variants: a modest ddG is more
                # rescuable than a catastrophic one. Hence a peak, not a ramp.
                stab_term = float(np.exp(-((v["ddg_pred"] - 1.9) ** 2) / 3.0) * 2.0 - 1.0)
                logit = (
                    self.w_pair * pair_term
                    + self.w_promisc * promisc_term
                    + self.w_stability * stab_term
                    + rng.normal(0, self.noise_sd)
                    - 0.85  # intercept -> realistic ~25-30% prevalence
                )
                p = 1.0 / (1.0 + np.exp(-logit))
                oracle.append(p)
                rows.append(
                    {
                        "variant_id": variant_id,
                        "compound_id": str(compound_id),
                        "label": int(rng.random() < p),
                        "assay": str(rng.choice(["enzyme_activity", "trafficking", "thermal_shift"])),
                        "source": "SYNTHETIC:foldrescue.data.SyntheticCohortAdapter",
                    }
                )
        pairs = pd.DataFrame(rows).reset_index(drop=True)
        weights = {
            "w_pair": self.w_pair,
            "w_promisc": self.w_promisc,
            "w_stability": self.w_stability,
            "noise_sd": self.noise_sd,
            # The quantity the attribution analysis tries to recover: how much
            # of the explainable logit variance is genuine pair interaction.
            "pair_signal_share": self.w_pair**2
            / (self.w_pair**2 + self.w_promisc**2 + self.w_stability**2),
        }
        return pairs, weights, np.asarray(oracle, dtype=float)


# ---------------------------------------------------------------------------
# Real benchmark adapter
# ---------------------------------------------------------------------------


@dataclass
class AmenabenchAdapter:
    """Read a curated benchmark in the documented schema.

    ``root`` should contain ``variants.csv``, ``compounds.csv``, ``pairs.csv``
    and optionally ``splits.json``. When files are absent the error names the
    exact paths checked and points at the schema, because the most likely
    reason for failure is that the benchmark has not been built yet.
    """

    root: Path
    name: str = "amenabench"

    def load(self) -> Cohort:
        root = Path(self.root)
        needed = {n: root / f"{n}.csv" for n in ("variants", "compounds", "pairs")}
        absent = [str(p) for p in needed.values() if not p.exists()]
        if absent:
            raise FileNotFoundError(
                "AmenabenchAdapter found no benchmark at "
                f"{root}. Missing: {absent}. The expected schema is documented in "
                "foldrescue/data.py (module docstring) and docs/data_contract.md. "
                "If you have not curated a benchmark yet, run with "
                "--cohort synthetic to exercise the pipeline on generated data "
                "(clearly labelled as such), or point --benchmark-root at your "
                "curated CSVs."
            )

        variants = pd.read_csv(needed["variants"], index_col=0)
        variants.index.name = "variant_id"
        compounds = pd.read_csv(needed["compounds"])
        compounds.index = pd.Index(compounds["compound_id"], name="compound_id")
        pairs = pd.read_csv(needed["pairs"])

        cohort = Cohort(
            variants=variants,
            compounds=compounds,
            pairs=pairs,
            synthetic=False,
            ground_truth_weights=None,
            provenance=f"AmenabenchAdapter({root})",
        )
        splits_path = root / "splits.json"
        if splits_path.exists():
            import json

            raw = json.loads(splits_path.read_text())
            cohort.splits = {
                name: {part: np.asarray(idx, dtype=int) for part, idx in parts.items()}
                for name, parts in raw.items()
            }
            _log.info("loaded %d predefined splits from %s", len(cohort.splits), splits_path)
        else:
            _log.warning(
                "no splits.json at %s; deriving splits locally. Predefined splits "
                "are strongly preferred for a benchmark -- locally derived splits "
                "are not comparable across papers.",
                splits_path,
            )
            cohort.splits = build_splits(cohort, seed=0)
        return cohort


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _scaffold_key(smiles: str) -> str:
    """Bemis-Murcko scaffold hash, falling back to a SMILES hash.

    Scaffold splitting answers a different question from random splitting:
    "does this generalise to a chemotype I have not seen?" Random splits over
    congeneric series flatter a model badly, because the test set is full of
    near-twins of the training set.
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
        from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            smi = Chem.MolToSmiles(scaffold)
            if smi:
                return smi
    except Exception:  # pragma: no cover - rdkit absent or exotic molecule
        pass
    return hashlib.sha1(smiles.encode()).hexdigest()[:12]


def _three_way(
    rng: np.random.Generator, idx: np.ndarray, calib_frac: float, test_frac: float
) -> dict[str, np.ndarray]:
    """Shuffle and cut an index into train/calib/test."""
    idx = np.asarray(idx, dtype=int)
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(round(n * test_frac))
    n_calib = int(round(n * calib_frac))
    return {
        "test": idx[:n_test],
        "calib": idx[n_test : n_test + n_calib],
        "train": idx[n_test + n_calib :],
    }


def _grouped(
    rng: np.random.Generator, groups: pd.Series, calib_frac: float, test_frac: float
) -> dict[str, np.ndarray]:
    """Split by *group*, so no group appears in two partitions.

    This is the only honest way to answer "unseen gene" / "unseen chemotype"
    questions. Splitting rows while a gene straddles train and test leaks the
    gene's idiosyncrasies and inflates every metric downstream.
    """
    uniq = np.array(sorted(groups.unique()))
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(n * test_frac)))
    n_calib = max(1, int(round(n * calib_frac))) if n - n_test > 1 else 0
    test_g = set(uniq[:n_test])
    calib_g = set(uniq[n_test : n_test + n_calib])
    pos = np.arange(len(groups))
    return {
        "test": pos[groups.isin(test_g).to_numpy()],
        "calib": pos[groups.isin(calib_g).to_numpy()],
        "train": pos[(~groups.isin(test_g | calib_g)).to_numpy()],
    }


def build_splits(
    cohort: Cohort,
    *,
    seed: int = 0,
    calib_frac: float = 0.15,
    test_frac: float = 0.20,
) -> dict[str, dict[str, np.ndarray]]:
    """Build the four named splits.

    Every split is three-way (train / calib / test). The calibration partition
    is not optional: temperature scaling or isotonic regression fitted on the
    test set reports a calibration error that is a fit statistic, not a
    measurement.
    """
    pairs = cohort.pairs
    joined = pairs.join(cohort.variants[["gene"]], on="variant_id")
    smiles_by_id = cohort.compounds["smiles"].to_dict()
    scaffolds = pairs["compound_id"].map(lambda c: _scaffold_key(smiles_by_id[c]))

    return {
        "random": _three_way(
            np.random.default_rng(seed), np.arange(len(pairs)), calib_frac, test_frac
        ),
        "gene_holdout": _grouped(
            np.random.default_rng(seed + 1), joined["gene"], calib_frac, test_frac
        ),
        "compound_holdout": _grouped(
            np.random.default_rng(seed + 2), pairs["compound_id"], calib_frac, test_frac
        ),
        "scaffold": _grouped(np.random.default_rng(seed + 3), scaffolds, calib_frac, test_frac),
    }


def holdout_gene_split(cohort: Cohort, gene: str, *, seed: int = 0, calib_frac: float = 0.18) -> dict[str, np.ndarray]:
    """Split that holds out one named gene entirely as the test set.

    Used for the prospective notebook: to ask "what would we predict for gene
    X having never seen it?", X must be absent from both train and calib.
    """
    joined = cohort.pairs.join(cohort.variants[["gene"]], on="variant_id")
    is_gene = (joined["gene"] == gene).to_numpy()
    if not is_gene.any():
        available = sorted(joined["gene"].unique())
        raise ValueError(f"gene {gene!r} not present in cohort; available: {available}")
    pos = np.arange(len(joined))
    rest = pos[~is_gene]
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    n_calib = int(round(len(rest) * calib_frac))
    return {"test": pos[is_gene], "calib": rest[:n_calib], "train": rest[n_calib:]}


def load_cohort(
    source: str = "synthetic",
    *,
    benchmark_root: Path | str | None = None,
    seed: int = 0,
    **kwargs: object,
) -> Cohort:
    """Front door for the data layer.

    ``source="synthetic"`` generates; anything else is treated as a benchmark
    root path (or the literal ``"amenabench"`` with ``benchmark_root`` set).
    """
    if source == "synthetic":
        return SyntheticCohortAdapter(seed=seed, **kwargs).load()  # type: ignore[arg-type]
    root = benchmark_root if benchmark_root is not None else source
    return AmenabenchAdapter(root=Path(root)).load()
