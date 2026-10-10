"""Applicability domain: deciding when a prediction should not be trusted.

THE PROBLEM
===========

A trained model will return a number for any input you hand it. Ask a model
trained on iminosugars about a macrocyclic peptide and it will answer 0.73
with no more hesitation than for a molecule it saw a hundred times. That
number is not a prediction; it is an extrapolation wearing a prediction's
clothes.

The applicability domain (AD) is the region of input space where a model's
error rate has actually been estimated. The concept is old and well-settled in
QSAR -- it is a regulatory requirement under the OECD's 2007 validation
principles for (Q)SAR models, principle 3 of five, precisely because
regulators were being handed confident predictions about chemicals unlike
anything in the training set. Netzeva et al. (2005) is the standard reference
for the definition.

foldrescue checks two axes independently, because a pair can be novel on
either side:

**Compound axis -- Tanimoto distance to the nearest training compound.**
The Tanimoto (Jaccard) coefficient on binary fingerprints is
``|A and B| / |A or B|``: shared substructures over total substructures. It is
the default chemical similarity measure because it is cheap, bounded, and
behaves sensibly on sparse binary vectors. Distance is ``1 - T``. The
conventional "similar enough" line is T >= 0.4 on ECFP4 (Patterson et al.
1996 on the similarity principle; the 0.35-0.4 band is where enrichment in
virtual screening falls apart), so the default threshold is a distance of
0.60. It is a *convention*, not a law of nature, and it is exposed as a
parameter.

**Protein axis -- sequence separation to the nearest trained position.**
A variant at residue 412 of a protein whose nearest training example is at
residue 180 is a different structural context, whatever the stability
predictor says. The check is per-gene: a position is in-domain if some
training variant in *the same gene* lies within N residues. Genes absent from
training entirely are out of domain by construction -- which is the honest
answer for a prospective query on a new gene, and is exactly the situation the
GLB1 holdout notebook puts itself in deliberately.

A query is out of domain if **either** axis fails. The CLI will not rank
out-of-domain pairs alongside in-domain ones without ``--include-ood``,
because sorting them into one list is how an extrapolation ends up at the top
of somebody's assay queue.

WHAT THIS IS NOT
----------------
An in-domain flag is not a correctness guarantee. It says the query resembles
data on which error was measured. The model can still be wrong -- just wrong
at the rate the benchmark reported, rather than at an unknown rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Cohort
from .utils import check_resources, get_logger, iter_chunks

__all__ = [
    "DomainVerdict",
    "ApplicabilityDomain",
    "tanimoto_distance_matrix",
    "DEFAULT_TANIMOTO_DISTANCE",
    "DEFAULT_RESIDUE_WINDOW",
]

_log = get_logger(__name__)

# 1 - 0.40: the similarity level below which ECFP4 nearest-neighbour inference
# is not empirically supported. See module docstring.
DEFAULT_TANIMOTO_DISTANCE = 0.60

# Residues. ~25 is roughly a secondary-structure element plus flanks: close
# enough that the local fold context is plausibly shared.
DEFAULT_RESIDUE_WINDOW = 25


@dataclass
class DomainVerdict:
    """Per-query domain decision, with the evidence that produced it.

    The evidence fields are not decoration -- a user told "out of domain"
    immediately wants to know *which* axis failed and *what* the nearest
    training example was, and a flag without that is unactionable.
    """

    out_of_domain: np.ndarray  # bool, (n,)
    compound_ood: np.ndarray  # bool, (n,)
    protein_ood: np.ndarray  # bool, (n,)
    nearest_compound: np.ndarray  # str, (n,)
    nearest_compound_distance: np.ndarray  # float, (n,)
    nearest_variant: np.ndarray  # str, (n,)
    nearest_residue_gap: np.ndarray  # float, (n,) -- inf when gene unseen
    gene_unseen: np.ndarray  # bool, (n,)
    thresholds: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.out_of_domain)

    @property
    def n_ood(self) -> int:
        return int(self.out_of_domain.sum())

    def reasons(self) -> list[str]:
        """Human-readable reason per query; empty string when in domain."""
        out = []
        for i in range(len(self)):
            if not self.out_of_domain[i]:
                out.append("")
                continue
            bits = []
            if self.compound_ood[i]:
                bits.append(
                    f"nearest training compound {self.nearest_compound[i]} at Tanimoto "
                    f"distance {self.nearest_compound_distance[i]:.3f} "
                    f"(> {self.thresholds.get('tanimoto_distance', float('nan')):.2f})"
                )
            if self.gene_unseen[i]:
                bits.append("gene absent from training data")
            elif self.protein_ood[i]:
                bits.append(
                    f"nearest trained position {self.nearest_residue_gap[i]:.0f} residues away "
                    f"(> {self.thresholds.get('residue_window', float('nan')):.0f})"
                )
            out.append("; ".join(bits))
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "out_of_domain": self.out_of_domain,
                "compound_ood": self.compound_ood,
                "protein_ood": self.protein_ood,
                "gene_unseen": self.gene_unseen,
                "nearest_training_neighbour": self.nearest_compound,
                "nearest_compound_distance": self.nearest_compound_distance,
                "nearest_training_variant": self.nearest_variant,
                "nearest_residue_gap": self.nearest_residue_gap,
                "ood_reason": self.reasons(),
            }
        )


def _morgan_bits(smiles_series: pd.Series, *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Binary Morgan fingerprints as a dense bool array.

    Binary, not counts: Tanimoto as defined above is a set operation. The
    count-based variant (MinMax) is a different coefficient, and mixing the two
    silently is a classic way to get similarity numbers that do not match
    anybody else's. Wider than the modelling fingerprint (2048 vs 1024) because
    here we want fidelity of the similarity estimate, not model compactness,
    and bit collisions inflate apparent similarity.
    """
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows, failures = [], []
    for cid, smi in smiles_series.items():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            failures.append(str(cid))
            rows.append(np.zeros(n_bits, dtype=bool))
            continue
        rows.append(np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=bool))
    if failures:
        # An all-zero fingerprint has Tanimoto 0 with everything, so it would
        # be flagged OOD -- the safe direction, but the user must be told.
        _log.warning(
            "could not parse %d SMILES for domain analysis (%s); these are "
            "treated as maximally dissimilar and will be flagged out-of-domain",
            len(failures),
            failures[:5],
        )
    return np.vstack(rows)


def tanimoto_distance_matrix(
    query_bits: np.ndarray,
    train_bits: np.ndarray,
    *,
    memory_fraction: float = 0.5,
) -> np.ndarray:
    """Pairwise Tanimoto *distance* (1 - similarity), shape (n_query, n_train).

    Computed as ``1 - inter / (|q| + |t| - inter)`` where ``inter`` is the
    bitwise-AND popcount, obtained as a matrix product of the bool arrays cast
    to a small integer type. The product is the expensive step, so it goes
    through the resource guard and is chunked over queries -- a 10^5 x 10^4
    float64 distance matrix is 8 GB, which is precisely the allocation the
    guard exists to stop.
    """
    q = np.asarray(query_bits, dtype=bool)
    t = np.asarray(train_bits, dtype=bool)
    if q.ndim != 2 or t.ndim != 2 or q.shape[1] != t.shape[1]:
        raise ValueError(f"fingerprint width mismatch: {q.shape} vs {t.shape}")

    plan = check_resources(
        q.shape[0],
        t.shape[0],
        dtype=np.float64,
        memory_fraction=memory_fraction,
        label=f"Tanimoto distance matrix ({q.shape[0]} x {t.shape[0]})",
    )
    out = np.empty((q.shape[0], t.shape[0]), dtype=plan.dtype)
    t_small = t.astype(np.int16)
    t_pop = t_small.sum(axis=1).astype(plan.dtype)

    for sl in iter_chunks(q.shape[0], plan):
        qc = q[sl].astype(np.int16)
        inter = (qc @ t_small.T).astype(plan.dtype)
        union = qc.sum(axis=1).astype(plan.dtype)[:, None] + t_pop[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[sl] = 1.0 - sim
    return out


class ApplicabilityDomain:
    """Fit on a training cohort slice; judge arbitrary query pairs.

    Usage::

        ad = ApplicabilityDomain().fit(cohort, train_rows)
        verdict = ad.judge(query_pairs, cohort)   # or a query cohort
    """

    def __init__(
        self,
        *,
        tanimoto_distance: float = DEFAULT_TANIMOTO_DISTANCE,
        residue_window: int = DEFAULT_RESIDUE_WINDOW,
        n_bits: int = 2048,
        radius: int = 2,
    ) -> None:
        if not 0.0 <= tanimoto_distance <= 1.0:
            raise ValueError(f"tanimoto_distance must be in [0, 1], got {tanimoto_distance}")
        if residue_window < 0:
            raise ValueError(f"residue_window must be >= 0, got {residue_window}")
        self.tanimoto_distance = float(tanimoto_distance)
        self.residue_window = int(residue_window)
        self.n_bits = int(n_bits)
        self.radius = int(radius)

        self.train_compound_ids_: np.ndarray = np.array([], dtype=object)
        self.train_bits_: np.ndarray = np.zeros((0, 0), dtype=bool)
        self.train_positions_: dict[str, np.ndarray] = {}
        self.train_variant_ids_: dict[str, np.ndarray] = {}
        self.fitted_ = False

    def fit(self, cohort: Cohort, train_rows: np.ndarray) -> ApplicabilityDomain:
        """Record what the model actually saw.

        ``train_rows`` indexes ``cohort.pairs`` -- the *training* partition
        only. Fitting the domain on the whole cohort would be the subtle
        version of test-set leakage: the domain would certify queries the model
        never trained on.
        """
        rows = np.asarray(train_rows, dtype=int)
        if len(rows) == 0:
            raise ValueError("cannot fit an applicability domain on zero training rows")
        train_pairs = cohort.pairs.iloc[rows]

        seen_compounds = pd.Index(sorted(set(train_pairs["compound_id"])))
        self.train_compound_ids_ = seen_compounds.to_numpy(dtype=object)
        self.train_bits_ = _morgan_bits(
            cohort.compounds.loc[seen_compounds, "smiles"],
            radius=self.radius,
            n_bits=self.n_bits,
        )

        seen_variants = sorted(set(train_pairs["variant_id"]))
        vinfo = cohort.variants.loc[seen_variants, ["gene", "position"]]
        self.train_positions_, self.train_variant_ids_ = {}, {}
        for gene, sub in vinfo.groupby("gene", observed=True):
            order = np.argsort(sub["position"].to_numpy())
            self.train_positions_[str(gene)] = sub["position"].to_numpy()[order]
            self.train_variant_ids_[str(gene)] = sub.index.to_numpy(dtype=object)[order]

        self.fitted_ = True
        _log.info(
            "applicability domain fitted on %d training pairs: %d compounds, "
            "%d variants across %d genes (thresholds: Tanimoto distance <= %.2f, "
            "residue window <= %d)",
            len(rows),
            len(seen_compounds),
            len(seen_variants),
            len(self.train_positions_),
            self.tanimoto_distance,
            self.residue_window,
        )
        return self

    def judge(
        self,
        query_pairs: pd.DataFrame,
        cohort: Cohort,
        *,
        memory_fraction: float = 0.5,
    ) -> DomainVerdict:
        """Judge query pairs against the fitted domain.

        ``query_pairs`` needs ``variant_id`` and ``compound_id``; both must
        resolve in ``cohort``. ``cohort`` may be the training cohort or a
        separate query cohort -- it is only used to look up SMILES and
        positions.
        """
        if not self.fitted_:
            raise RuntimeError("ApplicabilityDomain.fit must be called before judge")
        n = len(query_pairs)
        if n == 0:
            raise ValueError("no query pairs to judge")

        # --- compound axis: distance to nearest *training* compound -------
        q_compounds = pd.Index(sorted(set(query_pairs["compound_id"].astype(str))))
        q_bits = _morgan_bits(
            cohort.compounds.loc[q_compounds, "smiles"], radius=self.radius, n_bits=self.n_bits
        )
        dist = tanimoto_distance_matrix(q_bits, self.train_bits_, memory_fraction=memory_fraction)
        nn_idx = np.argmin(dist, axis=1)
        nn_dist = dist[np.arange(len(q_compounds)), nn_idx]
        per_compound_nn = {
            cid: (str(self.train_compound_ids_[j]), float(d))
            # strict=True: these three come from one argmin over one matrix,
            # so a length mismatch is a bug worth raising rather than truncating.
            for cid, j, d in zip(q_compounds, nn_idx, nn_dist, strict=True)
        }

        # --- protein axis: residue gap within the same gene ---------------
        q_variants = pd.Index(sorted(set(query_pairs["variant_id"].astype(str))))
        vinfo = cohort.variants.loc[q_variants, ["gene", "position"]]
        per_variant_nn: dict[str, tuple[str, float, bool]] = {}
        for vid, row in vinfo.iterrows():
            gene, pos = str(row["gene"]), float(row["position"])
            positions = self.train_positions_.get(gene)
            if positions is None or len(positions) == 0:
                per_variant_nn[str(vid)] = ("", float("inf"), True)
                continue
            # Positions are sorted at fit time, so the nearest is adjacent to
            # the insertion point -- O(log n) rather than O(n) per query.
            k = int(np.searchsorted(positions, pos))
            cands = [c for c in (k - 1, k) if 0 <= c < len(positions)]
            best = min(cands, key=lambda c: abs(positions[c] - pos))
            per_variant_nn[str(vid)] = (
                str(self.train_variant_ids_[gene][best]),
                float(abs(positions[best] - pos)),
                False,
            )

        nearest_compound = np.empty(n, dtype=object)
        nearest_compound_distance = np.empty(n, dtype=float)
        nearest_variant = np.empty(n, dtype=object)
        nearest_residue_gap = np.empty(n, dtype=float)
        gene_unseen = np.zeros(n, dtype=bool)

        for i, (vid, cid) in enumerate(
            zip(
                query_pairs["variant_id"].astype(str),
                query_pairs["compound_id"].astype(str),
                strict=True,
            )
        ):
            c_name, c_dist = per_compound_nn[cid]
            v_name, v_gap, unseen = per_variant_nn[vid]
            nearest_compound[i] = c_name
            nearest_compound_distance[i] = c_dist
            nearest_variant[i] = v_name
            nearest_residue_gap[i] = v_gap
            gene_unseen[i] = unseen

        compound_ood = nearest_compound_distance > self.tanimoto_distance
        protein_ood = nearest_residue_gap > self.residue_window  # inf > window -> True
        out_of_domain = compound_ood | protein_ood

        verdict = DomainVerdict(
            out_of_domain=out_of_domain,
            compound_ood=compound_ood,
            protein_ood=protein_ood,
            nearest_compound=nearest_compound,
            nearest_compound_distance=nearest_compound_distance,
            nearest_variant=nearest_variant,
            nearest_residue_gap=nearest_residue_gap,
            gene_unseen=gene_unseen,
            thresholds={
                "tanimoto_distance": self.tanimoto_distance,
                "residue_window": float(self.residue_window),
            },
        )
        _log.info(
            "domain verdict: %d/%d out of domain (%d compound-axis, %d protein-axis, "
            "%d unseen gene)",
            verdict.n_ood,
            n,
            int(compound_ood.sum()),
            int(protein_ood.sum()),
            int(gene_unseen.sum()),
        )
        return verdict
