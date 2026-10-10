"""Tests for the packaged UniProt sequence asset.

The asset is what makes the synthetic cohort's protein side real: positions
index a genuine sequence and ``wt_aa`` is the genuine residue there. If it
drifts out of sync with the generator, every ESM-2 embedding silently becomes
an embedding of the wrong context -- which produces confident vectors about
nothing and no error anywhere.
"""

from __future__ import annotations

import pytest

from foldrescue.data import _GENE_PANEL, AMINO_ACIDS, load_cohort, load_sequences


@pytest.fixture(scope="module")
def sequences():
    return load_sequences()


def test_asset_covers_every_gene_in_the_panel(sequences):
    assert set(sequences) == {gene for gene, _acc, _len in _GENE_PANEL}


def test_recorded_length_matches_the_sequence(sequences):
    for gene, record in sequences.items():
        assert len(record["sequence"]) == record["length"], gene


def test_accessions_match_the_panel(sequences):
    for gene, acc, _len in _GENE_PANEL:
        assert sequences[gene]["uniprot"] == acc, gene


def test_sequences_are_plausible_proteins(sequences):
    for gene, record in sequences.items():
        seq = record["sequence"]
        assert len(seq) > 300, f"{gene} is implausibly short"
        # Allow the rare non-standard codes UniProt uses, but they must be rare.
        odd = sum(1 for c in seq if c not in AMINO_ACIDS)
        assert odd / len(seq) < 0.01, f"{gene} has {odd} non-standard residues"
        assert seq[0] == "M", f"{gene} does not start with methionine"


def test_every_synthetic_variant_sits_on_its_real_residue():
    """The invariant the whole real-sequence change exists to establish."""
    cohort = load_cohort("synthetic", seed=0)
    sequences = load_sequences()
    mismatches = []
    for variant_id, v in cohort.variants.iterrows():
        seq = sequences[v["gene"]]["sequence"]
        assert 1 <= int(v["position"]) <= len(seq), f"{variant_id} is out of range"
        if seq[int(v["position"]) - 1] != v["wt_aa"]:
            mismatches.append(variant_id)
    assert not mismatches, (
        f"{len(mismatches)} variants carry a wt_aa that is not the real residue "
        f"at that position, e.g. {mismatches[:3]}. Variant windows built from "
        f"these would be fiction."
    )


def test_mutant_residue_always_differs_from_wild_type():
    cohort = load_cohort("synthetic", seed=0)
    same = cohort.variants[cohort.variants["wt_aa"] == cohort.variants["mut_aa"]]
    assert same.empty, f"{len(same)} variants substitute a residue for itself"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_invariant_holds_across_seeds(seed):
    cohort = load_cohort("synthetic", seed=seed)
    sequences = load_sequences()
    for _vid, v in cohort.variants.iterrows():
        assert sequences[v["gene"]]["sequence"][int(v["position"]) - 1] == v["wt_aa"]
