"""Tests for the applicability domain.

The specification requires: "the applicability-domain flag fires on a
deliberately out-of-distribution compound". That is
``test_flag_fires_on_deliberately_ood_compound``. The compound chosen is a
macrocyclic-peptide-like structure with no substructure overlap with the
iminosugar training set, so its nearest-neighbour Tanimoto distance is far
above any sensible threshold.

Equally important is the converse test: the flag must NOT fire on a close
analogue of a training compound. A domain check that flags everything is as
useless as one that flags nothing, and is harder to notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foldrescue.data import load_cohort
from foldrescue.domain import (
    DEFAULT_RESIDUE_WINDOW,
    DEFAULT_TANIMOTO_DISTANCE,
    ApplicabilityDomain,
    tanimoto_distance_matrix,
)

# A deliberately alien molecule: cyclic peptide backbone, no sugar, no
# iminosugar nitrogen in a polyol ring, nothing resembling the training set.
OOD_SMILES = "O=C1NC(Cc2ccccc2)C(=O)NC(CCCCN)C(=O)NC(C(C)C)C(=O)NC(Cc2c[nH]c3ccccc23)C(=O)N1"

# A near-twin of CMP_IMINO_01 (deoxynojirimycin-like): same ring, one extra
# methyl. Should be comfortably in domain.
IN_DOMAIN_ANALOGUE = "OC[C@H]1N(C)C[C@H](O)[C@@H](O)[C@@H]1O"


@pytest.fixture(scope="module")
def cohort():
    return load_cohort("synthetic", seed=0)


@pytest.fixture(scope="module")
def fitted_domain(cohort):
    return ApplicabilityDomain().fit(cohort, cohort.splits["random"]["train"])


def _query_cohort(cohort, extra_smiles: dict[str, str]):
    """Clone the cohort with extra compounds appended."""
    extra = pd.DataFrame(
        [{"compound_id": k, "smiles": v, "mechanism": "test"} for k, v in extra_smiles.items()]
    )
    extra.index = pd.Index(extra["compound_id"], name="compound_id")
    merged = pd.concat([cohort.compounds, extra])
    clone = type(cohort)(
        variants=cohort.variants,
        compounds=merged,
        pairs=cohort.pairs,
        splits=cohort.splits,
        synthetic=True,
        provenance="test",
    )
    return clone


def test_flag_fires_on_deliberately_ood_compound(cohort, fitted_domain):
    """REQUIRED BY SPEC: a deliberately out-of-distribution compound is flagged."""
    query_cohort = _query_cohort(cohort, {"OOD_CYCLIC_PEPTIDE": OOD_SMILES})
    variant_id = cohort.variants.index[0]
    pairs = pd.DataFrame(
        [{"variant_id": variant_id, "compound_id": "OOD_CYCLIC_PEPTIDE"}]
    )

    verdict = fitted_domain.judge(pairs, query_cohort)

    assert verdict.out_of_domain[0], "the alien compound was not flagged"
    assert verdict.compound_ood[0], "the compound axis should be the trigger"
    assert not verdict.protein_ood[0], "the variant is a training variant"
    assert verdict.nearest_compound_distance[0] > DEFAULT_TANIMOTO_DISTANCE
    # The reason must name the axis and the evidence, or it is unactionable.
    reason = verdict.reasons()[0]
    assert "Tanimoto" in reason
    assert verdict.nearest_compound[0] in reason


def test_flag_does_not_fire_on_a_close_analogue(cohort, fitted_domain):
    """The converse: a domain check that flags everything is useless."""
    query_cohort = _query_cohort(cohort, {"ANALOGUE": IN_DOMAIN_ANALOGUE})
    # Use a variant whose gene and position are in the training split.
    train_rows = cohort.splits["random"]["train"]
    variant_id = cohort.pairs.iloc[train_rows]["variant_id"].iloc[0]
    pairs = pd.DataFrame([{"variant_id": variant_id, "compound_id": "ANALOGUE"}])

    verdict = fitted_domain.judge(pairs, query_cohort)
    assert not verdict.compound_ood[0], (
        f"a close analogue was flagged (distance "
        f"{verdict.nearest_compound_distance[0]:.3f})"
    )
    assert verdict.reasons()[0] == ""


def test_training_compounds_are_at_zero_distance(cohort, fitted_domain):
    """Sanity: a compound in the training set is its own nearest neighbour."""
    train_rows = cohort.splits["random"]["train"]
    train_pairs = cohort.pairs.iloc[train_rows]
    cid = train_pairs["compound_id"].iloc[0]
    vid = train_pairs["variant_id"].iloc[0]
    verdict = fitted_domain.judge(
        pd.DataFrame([{"variant_id": vid, "compound_id": cid}]), cohort
    )
    assert verdict.nearest_compound_distance[0] == pytest.approx(0.0, abs=1e-9)
    assert verdict.nearest_compound[0] == cid
    assert not verdict.out_of_domain[0]


def test_unseen_gene_is_out_of_domain_by_construction(cohort):
    """A prospective query on a never-trained gene must be flagged.

    This is the GLB1-holdout situation: it is the honest answer, and the
    notebook relies on it.
    """
    gene = str(cohort.variants["gene"].iloc[0])
    joined = cohort.pairs.join(cohort.variants[["gene"]], on="variant_id")
    train_rows = np.flatnonzero((joined["gene"] != gene).to_numpy())
    ad = ApplicabilityDomain().fit(cohort, train_rows)

    held_out = cohort.variants.index[cohort.variants["gene"] == gene][0]
    cid = cohort.pairs["compound_id"].iloc[0]
    verdict = ad.judge(
        pd.DataFrame([{"variant_id": held_out, "compound_id": cid}]), cohort
    )
    assert verdict.gene_unseen[0]
    assert verdict.protein_ood[0]
    assert verdict.out_of_domain[0]
    assert np.isinf(verdict.nearest_residue_gap[0])
    assert "gene absent" in verdict.reasons()[0]


def test_protein_axis_fires_on_a_distant_residue(cohort):
    """Same gene, but no trained variant within the residue window."""
    gene = str(cohort.variants["gene"].iloc[0])
    same_gene = cohort.variants[cohort.variants["gene"] == gene]
    low = same_gene[same_gene["position"] < 100]
    if low.empty:
        pytest.skip("synthetic cohort has no low-position variant for this gene")

    # Train only on low positions of this gene, then query a high position.
    keep_ids = set(low.index)
    train_rows = np.flatnonzero(cohort.pairs["variant_id"].isin(keep_ids).to_numpy())
    if len(train_rows) == 0:
        pytest.skip("no pairs for the selected variants")
    ad = ApplicabilityDomain(residue_window=10).fit(cohort, train_rows)

    far = same_gene[same_gene["position"] > low["position"].max() + 200]
    if far.empty:
        pytest.skip("no sufficiently distant variant in this gene")
    verdict = ad.judge(
        pd.DataFrame(
            [{"variant_id": far.index[0], "compound_id": cohort.pairs["compound_id"].iloc[0]}]
        ),
        cohort,
    )
    assert verdict.protein_ood[0]
    assert not verdict.gene_unseen[0]
    assert verdict.nearest_residue_gap[0] > 10
    assert "residues away" in verdict.reasons()[0]


def test_either_axis_alone_triggers_ood(cohort, fitted_domain):
    """OOD is a disjunction: novel chemistry OR novel position is enough."""
    query_cohort = _query_cohort(cohort, {"OOD_CYCLIC_PEPTIDE": OOD_SMILES})
    train_pairs = cohort.pairs.iloc[cohort.splits["random"]["train"]]
    vid = train_pairs["variant_id"].iloc[0]
    cid = train_pairs["compound_id"].iloc[0]
    pairs = pd.DataFrame(
        [
            {"variant_id": vid, "compound_id": cid},  # both fine
            {"variant_id": vid, "compound_id": "OOD_CYCLIC_PEPTIDE"},  # compound novel
        ]
    )
    verdict = fitted_domain.judge(pairs, query_cohort)
    assert not verdict.out_of_domain[0]
    assert verdict.out_of_domain[1]


# --- Tanimoto mechanics ----------------------------------------------------


def test_tanimoto_identical_vectors_are_distance_zero():
    bits = np.array([[True, False, True, True]])
    d = tanimoto_distance_matrix(bits, bits)
    assert d.shape == (1, 1)
    assert d[0, 0] == pytest.approx(0.0)


def test_tanimoto_disjoint_vectors_are_distance_one():
    a = np.array([[True, True, False, False]])
    b = np.array([[False, False, True, True]])
    assert tanimoto_distance_matrix(a, b)[0, 0] == pytest.approx(1.0)


def test_tanimoto_known_value():
    """|A and B| = 1, |A or B| = 3 -> similarity 1/3, distance 2/3."""
    a = np.array([[True, True, False]])
    b = np.array([[True, False, True]])
    assert tanimoto_distance_matrix(a, b)[0, 0] == pytest.approx(2.0 / 3.0)


def test_tanimoto_all_zero_vectors_do_not_divide_by_zero():
    z = np.zeros((1, 8), dtype=bool)
    d = tanimoto_distance_matrix(z, z)
    assert np.isfinite(d).all()
    assert d[0, 0] == pytest.approx(1.0), "empty-vs-empty has no shared bits"


def test_tanimoto_width_mismatch_raises():
    with pytest.raises(ValueError, match="width mismatch"):
        tanimoto_distance_matrix(np.zeros((1, 4), dtype=bool), np.zeros((1, 8), dtype=bool))


def test_tanimoto_is_chunked_under_a_tight_budget():
    """The guard must actually be honoured, not merely consulted."""
    q = np.random.default_rng(0).random((200, 64)) > 0.5
    t = np.random.default_rng(1).random((150, 64)) > 0.5
    full = tanimoto_distance_matrix(q, t)
    tiny = tanimoto_distance_matrix(q, t, memory_fraction=1e-7)
    assert np.allclose(full, tiny), "chunked result must equal the unchunked one"


# --- construction / misuse -------------------------------------------------


def test_judge_before_fit_raises(cohort):
    with pytest.raises(RuntimeError, match="fit"):
        ApplicabilityDomain().judge(
            pd.DataFrame([{"variant_id": "x", "compound_id": "y"}]), cohort
        )


def test_fit_on_zero_rows_raises(cohort):
    with pytest.raises(ValueError, match="zero training rows"):
        ApplicabilityDomain().fit(cohort, np.array([], dtype=int))


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_invalid_threshold_rejected(bad):
    with pytest.raises(ValueError, match="tanimoto_distance"):
        ApplicabilityDomain(tanimoto_distance=bad)


def test_defaults_are_the_documented_conventions():
    ad = ApplicabilityDomain()
    assert ad.tanimoto_distance == DEFAULT_TANIMOTO_DISTANCE == 0.60
    assert ad.residue_window == DEFAULT_RESIDUE_WINDOW == 25


def test_verdict_frame_has_the_cli_columns(cohort, fitted_domain):
    train_pairs = cohort.pairs.iloc[cohort.splits["random"]["train"]].head(3)
    verdict = fitted_domain.judge(
        train_pairs[["variant_id", "compound_id"]].reset_index(drop=True), cohort
    )
    frame = verdict.to_frame()
    for col in ("out_of_domain", "nearest_training_neighbour", "ood_reason"):
        assert col in frame.columns
    assert len(frame) == 3
