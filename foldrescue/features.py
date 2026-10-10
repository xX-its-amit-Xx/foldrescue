"""Feature construction: three named blocks, assembled under the resource guard.

The feature matrix is deliberately organised into three *blocks*, each
answering a different question, because the ablation study in the cookbook
removes them one at a time and that is only meaningful if the blocks are
genuinely separable:

``stability``
    Properties of the variant as a folding problem: predicted ddG, solvent
    accessibility, residue depth, conservation, secondary structure, and the
    identity of the amino-acid substitution. This is the block a conventional
    stability predictor would give you, and on its own it cannot represent the
    compound at all -- so a model using only this block is forced to predict
    the same answer for every compound tested against a variant. That is the
    point: it bounds how much of the task is "is this variant destabilised?".

``structural``
    Where the variant sits relative to the chaperone binding pocket:
    CA-to-pocket distance, pocket volume change on mutation, and whether the
    residue lies in a loop contacting the ligand. This is the Tier 3 block.

``ligand``
    The compound: ECFP4 Morgan fingerprint (folded, counts) plus interpretable
    RDKit physicochemical descriptors. Fingerprint bits are opaque
    individually but the descriptors are not, which matters for attribution.

Interaction terms are added separately (``interaction`` block) because a
gradient-boosted tree can represent interactions implicitly, but only if the
relevant pair of columns is reachable in a single tree; a handful of explicit
products makes the pair-specific signal learnable at small n.

WHY ECFP4 AND NOT SOMETHING FANCIER
-----------------------------------
Extended-connectivity fingerprints (Rogers & Hahn 2010, building on Morgan's
1965 graph-canonicalisation algorithm) encode each atom's circular
neighbourhood out to a radius, hash those substructures, and fold them into a
fixed-width vector. Radius 2 ("ECFP4" -- the 4 is the diameter) is the
default across cheminformatics because it captures functional groups and their
immediate context without diluting into near-unique substructures. For
~12-10^3 compounds it is a better-behaved representation than a learned
embedding, which is the honest reason Tier 1 is hard to beat here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Cohort
from .utils import ResourcePlan, check_resources, get_logger, iter_chunks

__all__ = [
    "FeatureSpec",
    "FeatureMatrix",
    "build_features",
    "BLOCK_NAMES",
    "RDKIT_DESCRIPTORS",
]

_log = get_logger(__name__)

BLOCK_NAMES = ("stability", "structural", "ligand", "interaction")

# Interpretable descriptors, chosen so that attribution output is readable by
# a medicinal chemist rather than only by the model. Each is a named RDKit
# descriptor function.
RDKIT_DESCRIPTORS: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "NumAromaticRings",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
)

# Kyte-Doolittle hydropathy and side-chain volume (A^3). Substitution effects
# depend far more on these physical properties than on residue identity, and
# encoding them lets the model generalise to substitutions it has not seen.
_KD_HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
_SIDECHAIN_VOL = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
    "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
    "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
    "W": 227.8, "Y": 193.6,
}
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}


@dataclass(frozen=True)
class FeatureSpec:
    """Which blocks to build and how wide the fingerprint is.

    Parameters
    ----------
    blocks:
        Subset of ``BLOCK_NAMES``. The ablation notebook drives this.
    ecfp_bits:
        Folded fingerprint width. 1024 is a deliberate compromise: wide enough
        that bit collisions are rare for drug-sized molecules, narrow enough
        that a tree model with a few hundred positives is not drowned.
    ecfp_radius:
        Morgan radius. 2 == ECFP4.
    ecfp_counts:
        Count-based rather than binary bits. Counts retain how *many* times a
        substructure occurs, which matters for repeated motifs like polyols.
    """

    blocks: tuple[str, ...] = BLOCK_NAMES
    ecfp_bits: int = 1024
    ecfp_radius: int = 2
    ecfp_counts: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.blocks) - set(BLOCK_NAMES)
        if unknown:
            raise ValueError(f"unknown feature blocks {sorted(unknown)}; valid: {BLOCK_NAMES}")
        if not self.blocks:
            raise ValueError("at least one feature block is required")
        if "interaction" in self.blocks and not {"stability", "ligand"} & set(self.blocks):
            raise ValueError(
                "the 'interaction' block multiplies stability/structural terms against "
                "ligand terms; it is meaningless without at least one of them."
            )


@dataclass
class FeatureMatrix:
    """A built feature matrix plus the metadata attribution needs.

    Attributes
    ----------
    X:
        ``(n_pairs, n_features)`` array in the dtype the resource guard chose.
    y:
        Labels, or ``None`` for an inference-time matrix.
    feature_names:
        Column names, aligned to ``X``.
    block_of:
        ``feature_name -> block name``. Permutation importance aggregates over
        this to produce per-block numbers for the ablation study.
    plan:
        The ``ResourcePlan`` under which this matrix was built, kept for the
        provenance trail.
    """

    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]
    block_of: dict[str, str]
    plan: ResourcePlan
    pair_index: pd.DataFrame = field(default_factory=pd.DataFrame)

    def block_columns(self, block: str) -> list[int]:
        return [i for i, n in enumerate(self.feature_names) if self.block_of[n] == block]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        counts = {b: len(self.block_columns(b)) for b in sorted(set(self.block_of.values()))}
        return f"FeatureMatrix(X={self.X.shape}, dtype={self.X.dtype}, blocks={counts})"


# ---------------------------------------------------------------------------
# Block builders. Each returns (array, names).
# ---------------------------------------------------------------------------


def _stability_block(variants: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Variant-level folding features.

    Includes the *change* in hydropathy, volume and charge on substitution
    rather than only the endpoint identities, because the perturbation is what
    destabilises a fold -- swapping one buried hydrophobe for another is mild;
    burying a charge is not.
    """
    wt, mut = variants["wt_aa"], variants["mut_aa"]
    cols: dict[str, np.ndarray] = {
        "ddg_pred": variants["ddg_pred"].to_numpy(float),
        # Rescuability is non-monotonic in ddG: a barely-destabilised protein
        # does not need a chaperone, a catastrophically misfolded one cannot be
        # saved by one. Give the model the squared term so it can find a peak.
        "ddg_pred_sq": variants["ddg_pred"].to_numpy(float) ** 2,
        "rsa": variants["rsa"].to_numpy(float),
        "residue_depth": variants["residue_depth"].to_numpy(float),
        "conservation": variants["conservation"].to_numpy(float),
        "position_frac": np.zeros(len(variants)),  # filled below, per gene
        "hydropathy_wt": wt.map(_KD_HYDROPATHY).to_numpy(float),
        "hydropathy_delta": (mut.map(_KD_HYDROPATHY) - wt.map(_KD_HYDROPATHY)).to_numpy(float),
        "volume_delta": (mut.map(_SIDECHAIN_VOL) - wt.map(_SIDECHAIN_VOL)).to_numpy(float),
        "charge_delta": (
            mut.map(_CHARGE).fillna(0.0) - wt.map(_CHARGE).fillna(0.0)
        ).to_numpy(float),
        "is_proline_introduced": (mut == "P").to_numpy(float),
        "is_glycine_lost": (wt == "G").to_numpy(float),
        "is_cysteine_lost": (wt == "C").to_numpy(float),
        "buried_charge_introduced": (
            (variants["rsa"] < 0.15).to_numpy() & mut.isin(["D", "E", "K", "R"]).to_numpy()
        ).astype(float),
    }
    # Fractional position within each gene, so position generalises across
    # proteins of different lengths.
    frac = np.zeros(len(variants))
    for _gene, idx in variants.groupby("gene", observed=True).groups.items():
        sub = variants.loc[idx, "position"].to_numpy(float)
        span = max(sub.max(), 1.0)
        frac[variants.index.get_indexer(idx)] = sub / span
    cols["position_frac"] = frac

    for ss in ("H", "E", "L"):
        cols[f"sec_struct_{ss}"] = (variants["sec_struct"] == ss).to_numpy(float)

    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


def _structural_block(variants: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Pocket-occupancy features (Tier 3).

    ``dist_to_pocket`` is the headline term: a pharmacological chaperone works
    by binding the folded state and shifting the folding equilibrium, so a
    variant that deforms the binding pocket itself is a different proposition
    from one that destabilises a distal domain. Exponential decay terms are
    included because contact effects fall off sharply with distance, and a tree
    splitting on raw angstroms needs many splits to approximate that.
    """
    dist = variants["dist_to_pocket"].to_numpy(float)
    cols: dict[str, np.ndarray] = {
        "dist_to_pocket": dist,
        "dist_to_pocket_inv": 1.0 / (1.0 + dist),
        "dist_decay_8A": np.exp(-dist / 8.0),
        "pocket_vol_delta": variants["pocket_vol_delta"].to_numpy(float),
        "pocket_vol_delta_abs": np.abs(variants["pocket_vol_delta"].to_numpy(float)),
        "contacts_ligand": variants["contacts_ligand"].to_numpy(float),
        "in_pocket_shell": (dist < 8.0).astype(float),
        # A loop residue contacting the ligand is the classic "plastic" site:
        # tolerant of substitution but able to reshape the pocket.
        "loop_and_contacting": (
            (variants["sec_struct"] == "L").to_numpy() & (variants["contacts_ligand"] == 1).to_numpy()
        ).astype(float),
    }
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


def _ligand_block(
    compounds: pd.DataFrame, spec: FeatureSpec
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """ECFP + RDKit descriptors, one row per compound.

    Returns the per-compound matrix and a ``compound_id -> row`` map, so the
    pair matrix is assembled by indexing rather than by recomputing a
    fingerprint per pair. Recomputing would be the obvious implementation and
    is catastrophically slow once the pair table is large.
    """
    try:
        from rdkit import Chem, RDLogger  # noqa: PLC0415
        from rdkit.Chem import (
            Descriptors,  # noqa: PLC0415
            rdFingerprintGenerator,  # noqa: PLC0415
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The ligand feature block requires RDKit (pip install rdkit). "
            "To run without it, drop the block: FeatureSpec(blocks=('stability', "
            "'structural')) or pass --no-ligand-block to the CLI. Note that a "
            "model without the ligand block CANNOT distinguish compounds and so "
            "cannot answer the pair question at all -- it is an ablation, not a "
            "configuration."
        ) from exc

    # RDKit is chatty about sanitisation on real-world SMILES; we surface
    # failures ourselves below with better context.
    RDLogger.DisableLog("rdApp.*")

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=spec.ecfp_radius, fpSize=spec.ecfp_bits
    )
    desc_fns = []
    for name in RDKIT_DESCRIPTORS:
        fn = getattr(Descriptors, name, None)
        if fn is None:
            _log.warning("RDKit descriptor %s unavailable in this build; skipping", name)
            continue
        desc_fns.append((name, fn))

    rows: list[np.ndarray] = []
    row_of: dict[str, int] = {}
    failures: list[str] = []
    for compound_id, smiles in compounds["smiles"].items():
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            failures.append(f"{compound_id} ({smiles!r})")
            continue
        if spec.ecfp_counts:
            fp = np.asarray(gen.GetCountFingerprintAsNumPy(mol), dtype=float)
        else:
            fp = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=float)
        desc = np.array([float(fn(mol)) for _, fn in desc_fns], dtype=float)
        # Non-finite descriptors (rare, e.g. logP on exotic valences) would
        # poison the matrix silently; zero them and say so.
        if not np.all(np.isfinite(desc)):
            bad = [n for (n, _), v in zip(desc_fns, desc, strict=True) if not np.isfinite(v)]
            _log.warning("non-finite descriptors for %s: %s -> zeroed", compound_id, bad)
            desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        row_of[str(compound_id)] = len(rows)
        rows.append(np.concatenate([fp, desc]))

    if failures:
        raise ValueError(
            f"RDKit could not parse {len(failures)} SMILES: {failures[:5]}. "
            "Fix or drop these rows; foldrescue refuses to silently featurise a "
            "molecule it could not read."
        )

    names = [f"ecfp_{i}" for i in range(spec.ecfp_bits)] + [f"desc_{n}" for n, _ in desc_fns]
    return np.vstack(rows), names, row_of


def _interaction_block(
    stab: np.ndarray,
    stab_names: Sequence[str],
    struct: np.ndarray | None,
    struct_names: Sequence[str],
    lig_desc: np.ndarray,
    lig_desc_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Explicit products of a few protein terms against a few ligand terms.

    Which products? Only those with a mechanistic story, because the
    combinatorial space is large and spending model capacity on noise at
    n~10^3 is how you get a model that looks good on random splits and
    collapses on a gene holdout.

    - ddG x MolWt / MolLogP: a bigger, greasier ligand buries more surface and
      can stabilise a more destabilised fold.
    - dist_to_pocket x TPSA: polar-surface mismatch matters most for ligands
      that must actually reach into the pocket.
    - rsa x FractionCSP3: solvent exposure against ligand three-dimensionality.
    """
    wanted: tuple[tuple[str, str], ...] = (
        ("ddg_pred", "desc_MolWt"),
        ("ddg_pred", "desc_MolLogP"),
        ("rsa", "desc_FractionCSP3"),
        ("conservation", "desc_TPSA"),
        ("dist_to_pocket", "desc_TPSA"),
        ("dist_decay_8A", "desc_MolWt"),
        ("pocket_vol_delta_abs", "desc_NumRotatableBonds"),
    )
    prot_lookup: dict[str, np.ndarray] = {n: stab[:, i] for i, n in enumerate(stab_names)}
    if struct is not None:
        prot_lookup.update({n: struct[:, i] for i, n in enumerate(struct_names)})
    lig_lookup = {n: lig_desc[:, i] for i, n in enumerate(lig_desc_names)}

    cols, names = [], []
    for p_name, l_name in wanted:
        if p_name in prot_lookup and l_name in lig_lookup:
            cols.append(prot_lookup[p_name] * lig_lookup[l_name])
            names.append(f"inter_{p_name}_x_{l_name}")
    if not cols:
        return np.zeros((len(stab), 0)), []
    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_features(
    cohort: Cohort,
    *,
    spec: FeatureSpec | None = None,
    rows: np.ndarray | None = None,
    pairs: pd.DataFrame | None = None,
    memory_fraction: float = 0.5,
    n_jobs: int | None = None,
) -> FeatureMatrix:
    """Assemble the pair feature matrix under the resource guard.

    Parameters
    ----------
    cohort:
        Source of variants and compounds.
    spec:
        Which blocks to build. Defaults to all of them.
    rows:
        Row indices into ``cohort.pairs`` to build. ``None`` means all.
    pairs:
        Build for an arbitrary pair table instead of ``cohort.pairs`` -- the
        inference path, where pairs are the user's query, not observations.
        Must have ``variant_id`` and ``compound_id`` columns.
    memory_fraction, n_jobs:
        Passed to ``check_resources``.

    The guard runs *before* the pair matrix is materialised, and the chunked
    path is actually used -- we fill a preallocated output array slice by
    slice. This is not decorative: the pair matrix is
    ``n_pairs x (ecfp_bits + ~40)``, which at 10^6 pairs and 1024 bits is
    ~8 GB in float64 and ~4 GB in float32.
    """
    spec = spec or FeatureSpec()
    pair_table = cohort.pairs if pairs is None else pairs
    if rows is not None:
        pair_table = pair_table.iloc[np.asarray(rows, dtype=int)]
    n_pairs = len(pair_table)
    if n_pairs == 0:
        raise ValueError("no pairs to featurise (empty selection)")

    missing_v = set(pair_table["variant_id"]) - set(cohort.variants.index)
    missing_c = set(pair_table["compound_id"]) - set(cohort.compounds.index)
    if missing_v or missing_c:
        raise KeyError(
            f"pair table references unknown ids -- variants: {sorted(missing_v)[:3]}, "
            f"compounds: {sorted(missing_c)[:3]}"
        )

    # --- per-entity blocks (small: one row per variant / per compound) ----
    stab_all, stab_names = _stability_block(cohort.variants)
    struct_all, struct_names = (
        _structural_block(cohort.variants) if "structural" in spec.blocks else (None, [])
    )

    lig_all: np.ndarray | None = None
    lig_names: list[str] = []
    lig_row_of: dict[str, int] = {}
    if "ligand" in spec.blocks or "interaction" in spec.blocks:
        lig_all, lig_names, lig_row_of = _ligand_block(cohort.compounds, spec)

    # --- decide the output width before allocating ------------------------
    block_of: dict[str, str] = {}
    out_names: list[str] = []
    if "stability" in spec.blocks:
        out_names += stab_names
        block_of.update(dict.fromkeys(stab_names, "stability"))
    if "structural" in spec.blocks:
        out_names += struct_names
        block_of.update(dict.fromkeys(struct_names, "structural"))
    if "ligand" in spec.blocks:
        out_names += lig_names
        block_of.update(dict.fromkeys(lig_names, "ligand"))

    desc_names = [n for n in lig_names if n.startswith("desc_")]
    inter_names: list[str] = []
    if "interaction" in spec.blocks and lig_all is not None:
        # Build names only, on a single probe row, to learn the width without
        # allocating the full matrix first.
        probe_desc = lig_all[:1, [lig_names.index(n) for n in desc_names]]
        _, inter_names = _interaction_block(
            stab_all[:1], stab_names,
            None if struct_all is None else struct_all[:1], struct_names,
            probe_desc, desc_names,
        )
        out_names += inter_names
        block_of.update(dict.fromkeys(inter_names, "interaction"))

    n_cols = len(out_names)
    plan = check_resources(
        n_pairs,
        n_cols,
        dtype=np.float64,
        n_jobs=n_jobs,
        memory_fraction=memory_fraction,
        label=f"pair feature matrix ({n_pairs} x {n_cols})",
    )

    X = np.empty((n_pairs, n_cols), dtype=plan.dtype)

    v_pos = cohort.variants.index.get_indexer(pair_table["variant_id"])
    c_rows = (
        np.asarray([lig_row_of[str(c)] for c in pair_table["compound_id"]], dtype=int)
        if lig_all is not None
        else None
    )
    desc_idx = [lig_names.index(n) for n in desc_names] if lig_all is not None else []

    # Honour the plan: fill in chunks so peak memory stays near the estimate.
    for sl in iter_chunks(n_pairs, plan):
        parts: list[np.ndarray] = []
        vp = v_pos[sl]
        if "stability" in spec.blocks:
            parts.append(stab_all[vp])
        if "structural" in spec.blocks and struct_all is not None:
            parts.append(struct_all[vp])
        if "ligand" in spec.blocks and lig_all is not None:
            parts.append(lig_all[c_rows[sl]])
        if inter_names and lig_all is not None:
            inter, _ = _interaction_block(
                stab_all[vp], stab_names,
                None if struct_all is None else struct_all[vp], struct_names,
                lig_all[c_rows[sl]][:, desc_idx], desc_names,
            )
            parts.append(inter)
        X[sl] = np.column_stack(parts).astype(plan.dtype, copy=False)

    y = pair_table["label"].to_numpy(int) if "label" in pair_table.columns else None
    _log.info(
        "built features: %s (%s)%s",
        X.shape,
        plan.dtype,
        " [chunked]" if plan.chunked else "",
    )
    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=out_names,
        block_of=block_of,
        plan=plan,
        pair_index=pair_table[["variant_id", "compound_id"]].reset_index(drop=True),
    )
