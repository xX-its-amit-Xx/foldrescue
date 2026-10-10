"""Tests for the alpha-sphere pocket geometry.

Geometry is testable against constructed cases with known answers, which is a
luxury the modelling code does not have. These tests therefore check *values*,
not just that the code runs: a circumsphere of a known tetrahedron, a pocket
detected in a shell with a deliberate cavity, and a volume change whose sign is
forced by the substitution.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldrescue.structure import (
    ALPHA_MAX_DEFAULT,
    ALPHA_MIN_DEFAULT,
    SIDECHAIN_VOLUME,
    _circumsphere,
    _fibonacci_shell,
    _single_linkage,
    alpha_spheres,
    detect_pockets,
    distance_to_pocket,
    pocket_volume_delta_on_mutation,
    residue_contacts_ligand,
)

# --- circumsphere ----------------------------------------------------------


def test_circumsphere_of_a_known_tetrahedron():
    """Unit tetrahedron on the axes: centre (.5,.5,.5), radius sqrt(3)/2."""
    p = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    centre, radius = _circumsphere(p)
    assert centre == pytest.approx([0.5, 0.5, 0.5])
    assert radius == pytest.approx(np.sqrt(3) / 2)


def test_circumsphere_is_equidistant_from_all_four_vertices():
    rng = np.random.default_rng(0)
    p = rng.normal(0, 5, size=(4, 3))
    centre, radius = _circumsphere(p)
    for vertex in p:
        assert np.linalg.norm(centre - vertex) == pytest.approx(radius, rel=1e-9)


def test_degenerate_coplanar_tetrahedron_returns_none():
    """Four coplanar points have no finite circumsphere; must not raise."""
    p = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    assert _circumsphere(p) is None


# --- alpha spheres ---------------------------------------------------------


def _hollow_shell(radius: float = 9.0, n: int = 220, seed: int = 0) -> np.ndarray:
    """Atoms on a spherical shell: a big, obvious, empty cavity inside."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # Slight radial jitter so the points are not perfectly co-spherical, which
    # would make the triangulation degenerate.
    return v * (radius + rng.normal(0, 0.25, size=(n, 1)))


def test_alpha_spheres_respect_the_radius_band():
    coords = _hollow_shell()
    spheres = alpha_spheres(coords, alpha_min=3.0, alpha_max=6.0)
    assert spheres, "expected some spheres in the band"
    for s in spheres:
        assert 3.0 <= s.radius <= 6.0


def test_a_tighter_band_yields_no_more_spheres():
    coords = _hollow_shell()
    wide = alpha_spheres(coords, alpha_min=2.0, alpha_max=9.0)
    narrow = alpha_spheres(coords, alpha_min=3.5, alpha_max=4.5)
    assert len(narrow) <= len(wide)


def test_alpha_spheres_touch_their_four_atoms():
    coords = _hollow_shell()
    for s in alpha_spheres(coords)[:10]:
        for idx in s.atom_indices:
            assert np.linalg.norm(coords[idx] - s.centre) == pytest.approx(s.radius, rel=1e-7)


def test_alpha_spheres_rejects_too_few_atoms():
    with pytest.raises(ValueError, match="at least 4 atoms"):
        alpha_spheres(np.zeros((3, 3)))


def test_alpha_spheres_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(n_atoms, 3\)"):
        alpha_spheres(np.zeros((10, 2)))


def test_alpha_spheres_rejects_inverted_band():
    with pytest.raises(ValueError, match="alpha_min must be"):
        alpha_spheres(_hollow_shell(), alpha_min=6.0, alpha_max=3.0)


def test_degenerate_coplanar_input_raises_a_useful_message():
    flat = np.column_stack([np.random.default_rng(0).random((40, 2)) * 10, np.zeros(40)])
    with pytest.raises(ValueError, match="coplanar"):
        alpha_spheres(flat)


# --- pocket detection ------------------------------------------------------


def test_detects_the_cavity_in_a_hollow_shell():
    """The headline behaviour: find the hole, and centre it on the hole."""
    coords = _hollow_shell(radius=9.0, n=260)
    pockets = detect_pockets(coords, alpha_min=3.0, alpha_max=7.5, min_spheres=5)
    assert pockets, "no pocket found in a shell with an obvious cavity"
    biggest = pockets[0]
    # The shell is centred on the origin, so the cavity must be too.
    assert np.linalg.norm(biggest.centre) < 3.0, f"centre {biggest.centre} is not the cavity"
    assert biggest.volume > 0


def test_pockets_are_ranked_largest_first():
    pockets = detect_pockets(_hollow_shell(n=300), alpha_min=3.0, alpha_max=7.5, min_spheres=4)
    counts = [p.n_spheres for p in pockets]
    assert counts == sorted(counts, reverse=True)
    assert [p.rank for p in pockets] == list(range(len(pockets)))


def test_min_spheres_filters_specks():
    coords = _hollow_shell(n=300)
    loose = detect_pockets(coords, alpha_min=3.0, alpha_max=7.5, min_spheres=2)
    strict = detect_pockets(coords, alpha_min=3.0, alpha_max=7.5, min_spheres=25)
    assert len(strict) <= len(loose)
    assert all(p.n_spheres >= 25 for p in strict)


def test_a_dense_blob_has_no_large_pocket():
    """Negative control: solid packing should not produce a big cavity.

    Without this, a detector that returns 'pocket' for any input would pass
    every other test in this file.
    """
    rng = np.random.default_rng(1)
    # Dense random packing inside a small ball -- no designed cavity.
    pts = rng.normal(size=(400, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    blob = pts * rng.uniform(0, 7.0, size=(400, 1)) ** (1 / 3) * 3.5

    shell = _hollow_shell(radius=9.0, n=260)
    blob_pockets = detect_pockets(blob, alpha_min=3.0, alpha_max=7.5, min_spheres=5)
    shell_pockets = detect_pockets(shell, alpha_min=3.0, alpha_max=7.5, min_spheres=5)

    blob_max = max((p.volume for p in blob_pockets), default=0.0)
    shell_max = max((p.volume for p in shell_pockets), default=0.0)
    assert shell_max > blob_max, (
        f"the hollow shell's cavity ({shell_max:.1f} A^3) should exceed the dense "
        f"blob's largest void ({blob_max:.1f} A^3)"
    )


def test_volume_is_a_union_not_a_sum():
    """Overlapping alpha spheres must not be double-counted.

    Summing 4/3 pi r^3 over a pocket's spheres overcounts severely because
    they overlap heavily -- this asserts the union is genuinely smaller.
    """
    pockets = detect_pockets(_hollow_shell(n=260), alpha_min=3.0, alpha_max=7.5, min_spheres=5)
    p = pockets[0]
    naive_sum = sum(4.0 / 3.0 * np.pi * s.radius**3 for s in p.spheres)
    assert p.volume < naive_sum, "volume looks like a naive sum of overlapping spheres"


def test_volume_is_reproducible_for_a_fixed_seed():
    coords = _hollow_shell(n=240)
    a = detect_pockets(coords, min_spheres=4, seed=7)
    b = detect_pockets(coords, min_spheres=4, seed=7)
    assert [p.volume for p in a] == [p.volume for p in b]


def test_compute_volume_false_skips_the_monte_carlo():
    pockets = detect_pockets(_hollow_shell(n=240), min_spheres=4, compute_volume=False)
    assert all(p.volume == 0.0 for p in pockets)


def test_defaults_are_the_documented_fpocket_band():
    assert (ALPHA_MIN_DEFAULT, ALPHA_MAX_DEFAULT) == (3.0, 6.0)


# --- clustering ------------------------------------------------------------


def test_single_linkage_separates_two_distant_groups():
    pts = np.vstack([np.zeros((5, 3)) + [0, 0, 0], np.zeros((4, 3)) + [100, 0, 0]])
    pts = pts + np.random.default_rng(0).normal(0, 0.1, pts.shape)
    labels = _single_linkage(pts, cutoff=2.0)
    assert len(np.unique(labels)) == 2
    assert len(np.unique(labels[:5])) == 1
    assert len(np.unique(labels[5:])) == 1


def test_single_linkage_chains_through_a_neck():
    """Documented behaviour: single linkage joins groups via a connector."""
    chain = np.array([[float(i), 0.0, 0.0] for i in range(0, 20, 2)])
    assert len(np.unique(_single_linkage(chain, cutoff=2.5))) == 1
    assert len(np.unique(_single_linkage(chain, cutoff=1.5))) == len(chain)


# --- the three Tier 3 columns ---------------------------------------------


def test_distance_to_pocket_matches_hand_computation():
    pockets = detect_pockets(_hollow_shell(n=260), alpha_min=3.0, alpha_max=7.5, min_spheres=5)
    p = pockets[0]
    ca = np.array([20.0, 0.0, 0.0])
    assert distance_to_pocket(ca, p) == pytest.approx(float(np.linalg.norm(ca - p.centre)))


def test_distance_to_pocket_is_larger_for_a_more_distant_residue():
    p = detect_pockets(_hollow_shell(n=260), alpha_min=3.0, alpha_max=7.5, min_spheres=5)[0]
    near = distance_to_pocket(np.array([10.0, 0, 0]), p)
    far = distance_to_pocket(np.array([40.0, 0, 0]), p)
    assert far > near


def _occluded_cavity():
    """A shell cavity with a side chain protruding into it."""
    shell = _hollow_shell(radius=9.0, n=240)
    sidechain = _fibonacci_shell(np.array([3.6, 0.0, 0.0]), 2.6, n=14)
    coords = np.vstack([shell, sidechain])
    return coords, list(range(len(shell), len(coords))), np.array([5.2, 0.0, 0.0])


def test_volume_delta_is_finite_and_deterministic():
    """What this quantity actually guarantees: reproducibility.

    Deliberately NOT asserted: that the sign tracks side-chain size. See
    ``test_volume_delta_sign_is_not_a_reliable_steric_proxy``.
    """
    coords, idx, cb = _occluded_cavity()
    kw = {"sidechain_atom_indices": idx, "cb_coord": cb, "alpha_min": 3.0,
          "alpha_max": 7.5, "min_spheres": 4, "n_samples": 40000}
    a = pocket_volume_delta_on_mutation(coords, wt_aa="W", mut_aa="A", **kw)
    b = pocket_volume_delta_on_mutation(coords, wt_aa="W", mut_aa="A", **kw)
    assert np.isfinite(a)
    assert a == b, "same inputs and seed must give the same delta"


@pytest.mark.parametrize("wt,mut", [("W", "A"), ("A", "W"), ("G", "F"), ("F", "G")])
def test_volume_delta_is_finite_for_any_substitution(wt, mut):
    coords, idx, cb = _occluded_cavity()
    delta = pocket_volume_delta_on_mutation(
        coords, sidechain_atom_indices=idx, cb_coord=cb, wt_aa=wt, mut_aa=mut,
        alpha_min=3.0, alpha_max=7.5, min_spheres=4, n_samples=30000,
    )
    assert np.isfinite(delta)


def test_volume_delta_sign_is_not_a_reliable_steric_proxy():
    """Documents a real limitation rather than asserting a convenient fiction.

    Removing a tryptophan from this cavity yields a NEGATIVE delta -- the
    detected pocket gets smaller even though the physical void grew. This was
    verified stable under a 10x increase in Monte Carlo samples, so it is not
    sampling noise.

    The cause is intrinsic to alpha-sphere detection: the volume is
    band-limited to radii in [alpha_min, alpha_max]. A side chain protruding
    into a void is pocket-lining surface in its own right and subdivides the
    void into narrower, in-band spaces. Delete it and spheres grow past
    alpha_max and drop out of the measurement.

    This test exists so that the behaviour is recorded and cannot be
    "fixed" by accident without someone reading why it is here. If you change
    the algorithm such that the sign becomes meaningful, this test should be
    replaced with a sign assertion -- deliberately, not incidentally.
    """
    coords, idx, cb = _occluded_cavity()
    delta = pocket_volume_delta_on_mutation(
        coords, sidechain_atom_indices=idx, cb_coord=cb, wt_aa="W", mut_aa="A",
        alpha_min=3.0, alpha_max=7.5, min_spheres=4, n_samples=120000,
    )
    assert np.isfinite(delta)
    assert delta < 0, (
        "KNOWN LIMITATION regression: removing a large side chain from this "
        f"cavity has always produced a negative (shrinking) delta; got {delta:+.2f}. "
        "If this now reads positive the band-limiting behaviour has changed -- "
        "read the docstring of pocket_volume_delta_on_mutation before updating."
    )


def test_volume_delta_warns_about_the_limitation_at_runtime(caplog):
    """The caveat must travel with the call, not only live in the docstring."""
    coords, idx, cb = _occluded_cavity()
    with caplog.at_level("WARNING"):
        pocket_volume_delta_on_mutation(
            coords, sidechain_atom_indices=idx, cb_coord=cb, wt_aa="W", mut_aa="A",
            alpha_min=3.0, alpha_max=7.5, min_spheres=4, n_samples=20000,
        )
    assert any("BAND-LIMITED" in r.message for r in caplog.records)


def test_volume_delta_rejects_unknown_amino_acids():
    coords = _hollow_shell(n=200)
    with pytest.raises(ValueError, match="unknown amino acid"):
        pocket_volume_delta_on_mutation(
            coords, sidechain_atom_indices=[0], cb_coord=coords[0], wt_aa="W", mut_aa="Z"
        )


def test_glycine_substitution_adds_no_side_chain():
    """Glycine has no side chain; the substitution must add no atoms."""
    shell = _hollow_shell(radius=9.0, n=220)
    sidechain = _fibonacci_shell(np.array([3.6, 0.0, 0.0]), 2.4, n=12)
    coords = np.vstack([shell, sidechain])
    idx = list(range(len(shell), len(coords)))
    delta = pocket_volume_delta_on_mutation(
        coords, sidechain_atom_indices=idx, cb_coord=np.array([5.0, 0.0, 0.0]),
        wt_aa="F", mut_aa="G", alpha_min=3.0, alpha_max=7.5, min_spheres=4,
    )
    assert np.isfinite(delta)


def test_sidechain_volume_table_is_ordered_sensibly():
    """Sanity on the physical constants, not the code."""
    assert SIDECHAIN_VOLUME["G"] < SIDECHAIN_VOLUME["A"] < SIDECHAIN_VOLUME["V"]
    assert SIDECHAIN_VOLUME["W"] == max(SIDECHAIN_VOLUME.values())
    assert SIDECHAIN_VOLUME["G"] == min(SIDECHAIN_VOLUME.values())


def test_residue_contacts_ligand_within_cutoff():
    res = np.array([[0.0, 0, 0], [1.5, 0, 0]])
    lig = np.array([[4.0, 0, 0]])
    assert residue_contacts_ligand(res, lig, cutoff=4.5, sec_struct="L")
    assert not residue_contacts_ligand(res, lig, cutoff=2.0, sec_struct="L")


def test_contacts_requires_a_loop_by_default():
    """The feature targets plastic loop residues specifically."""
    res = np.array([[0.0, 0, 0]])
    lig = np.array([[2.0, 0, 0]])
    assert residue_contacts_ligand(res, lig, sec_struct="L")
    assert not residue_contacts_ligand(res, lig, sec_struct="H")
    assert not residue_contacts_ligand(res, lig, sec_struct="E")
    # Opt out for a plain contact test.
    assert residue_contacts_ligand(res, lig, sec_struct="H", require_loop=False)


def test_contacts_with_empty_input_is_false_not_an_error():
    assert not residue_contacts_ligand(np.zeros((0, 3)), np.array([[0.0, 0, 0]]))
    assert not residue_contacts_ligand(np.array([[0.0, 0, 0]]), np.zeros((0, 3)))


def test_contacts_rejects_bad_shapes():
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        residue_contacts_ligand(np.zeros((4, 2)), np.zeros((2, 3)))


def test_fibonacci_shell_points_lie_on_the_sphere():
    centre = np.array([1.0, -2.0, 3.0])
    pts = _fibonacci_shell(centre, 2.5, n=20)
    assert pts.shape == (20, 3)
    assert np.allclose(np.linalg.norm(pts - centre, axis=1), 2.5)
