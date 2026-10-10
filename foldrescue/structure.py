"""Pocket geometry from atomic coordinates: a small alpha-sphere implementation.

WHAT AN ALPHA SPHERE IS, AND WHY POCKET DETECTION USES THEM
===========================================================

A binding pocket is a concavity — a region of empty space enclosed by protein
atoms. "Empty space enclosed by atoms" is awkward to define directly, so the
standard trick inverts it.

An **alpha sphere** is a sphere that touches exactly four atoms and contains no
atom inside it. Every such sphere is centred on a vertex of the **Voronoi
diagram** of the atom centres — equivalently, it is the circumsphere of a
**Delaunay** tetrahedron. (The Delaunay triangulation is the dual of the
Voronoi diagram, and its defining property is exactly the empty-circumsphere
condition, which is why the two problems are the same problem.) Computing the
Delaunay tetrahedra therefore hands you every candidate alpha sphere at once,
in O(n log n), with no search.

The radius then classifies the local geometry, which is the part that matters:

* **very small radius** — the four atoms are packed tightly around the sphere.
  This is the protein *interior*.
* **intermediate radius (~3–6 Å)** — a gap big enough for a water or a ligand
  fragment, still enclosed by protein on several sides. **This is a pocket.**
* **large radius** — the sphere escapes into bulk solvent. This is the
  *exterior*.

So pocket detection reduces to: build the Delaunay triangulation, keep the
tetrahedra whose circumsphere radius falls in the intermediate band, and
cluster the survivors into connected groups. Each cluster is a candidate
pocket, and its size ranks them.

This is the algorithm behind **fpocket** (Le Guilloux, Schmidtke & Tuffery
2009), itself building on Liang, Edelsbrunner & Woodward's alpha-shape theory
from the 1990s. ``fpocket`` is a compiled binary with its own scoring, filters
and druggability model; this module is a deliberately small, dependency-light
reimplementation of its geometric core. It is not a substitute for fpocket
where fpocket is available, and it says so in ``detect_pockets``.

WHAT THIS MODULE IS FOR
-----------------------
Producing the three Tier 3 columns of the data contract from real coordinates:

* ``dist_to_pocket``    — Cα-to-pocket-centroid distance, Å
* ``pocket_vol_delta``  — change in pocket volume when the residue is mutated
* ``contacts_ligand``   — residue in a loop contacting the ligand

Without coordinates these must be supplied in ``variants.csv``. The synthetic
cohort supplies them directly (it has no structures), so this module is
exercised by its own tests on constructed geometry rather than by the cookbook.

THE HONEST CAVEAT ON ``pocket_vol_delta``
-----------------------------------------
Properly, the volume change on mutation requires repacking the side chain and
relaxing the structure — a Rosetta/FoldX-scale job. What
``pocket_volume_delta_on_mutation`` does instead is a **side-chain volume
substitution**: it replaces the wild-type residue's side-chain atoms with a
sphere of the mutant's characteristic volume at the Cβ position and recomputes.
That captures the first-order steric effect (a Trp→Ala substitution opens the
pocket; Gly→Trp fills it) and nothing else. It does not model backbone shift,
rotamer choice, or induced fit. The docstring repeats this where it is
returned, because a number called "pocket volume change" invites more
confidence than this computation earns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import check_resources, get_logger

__all__ = [
    "AlphaSphere",
    "Pocket",
    "detect_pockets",
    "alpha_spheres",
    "distance_to_pocket",
    "pocket_volume_delta_on_mutation",
    "residue_contacts_ligand",
    "SIDECHAIN_VOLUME",
    "VDW_RADII",
]

_log = get_logger(__name__)

# Van der Waals radii, Å (Bondi 1964). Used to inflate atom centres when
# estimating occupied volume.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}
DEFAULT_RADIUS = 1.70

# Side-chain volumes, Å^3 (Zamyatnin 1972). Same table as features.py uses for
# the substitution-effect features; duplicated deliberately so this module
# stands alone as a geometry utility.
SIDECHAIN_VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
    "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
    "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
    "W": 227.8, "Y": 193.6,
}

# The intermediate radius band. Below MIN we are inside the packed core; above
# MAX the sphere has escaped to bulk solvent. These are fpocket's defaults and
# are exposed as parameters because they are conventions, not constants.
ALPHA_MIN_DEFAULT = 3.0
ALPHA_MAX_DEFAULT = 6.0


@dataclass(frozen=True)
class AlphaSphere:
    """One empty circumsphere of a Delaunay tetrahedron."""

    centre: np.ndarray  # (3,)
    radius: float
    atom_indices: tuple[int, int, int, int]


@dataclass
class Pocket:
    """A connected cluster of alpha spheres.

    Attributes
    ----------
    centre:
        Centroid of the member sphere centres. This is what
        ``dist_to_pocket`` measures to.
    spheres:
        Member alpha spheres.
    volume:
        Union volume of the member spheres, by Monte Carlo integration. Note
        that this is the volume of the *cavity*, not of the protein.
    rank:
        0 = largest pocket by sphere count. Pocket 0 is the usual guess for
        an orthosteric site, and that guess is wrong often enough that it
        should be checked against a known ligand before being trusted.
    """

    centre: np.ndarray
    spheres: list[AlphaSphere]
    volume: float = 0.0
    rank: int = 0
    atom_indices: frozenset[int] = field(default_factory=frozenset)

    @property
    def n_spheres(self) -> int:
        return len(self.spheres)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        c = np.round(self.centre, 2)
        return f"Pocket(rank={self.rank}, n_spheres={self.n_spheres}, volume={self.volume:.1f} A^3, centre={c})"


def _circumsphere(p: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Circumcentre and circumradius of a tetrahedron, or None if degenerate.

    Solves the linear system from the equal-distance condition
    ``|x - p0|^2 = |x - pi|^2``, which expands to
    ``2 (pi - p0) . x = |pi|^2 - |p0|^2``. Degenerate (coplanar) tetrahedra
    give a singular matrix and are skipped — they carry no volume and
    scipy's Delaunay can emit a few from near-coplanar atom quadruples.
    """
    p0 = p[0]
    a = 2.0 * (p[1:] - p0)
    b = np.sum(p[1:] ** 2, axis=1) - np.sum(p0**2)
    try:
        centre = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(centre)):
        return None
    return centre, float(np.linalg.norm(centre - p0))


def alpha_spheres(
    coords: np.ndarray,
    *,
    alpha_min: float = ALPHA_MIN_DEFAULT,
    alpha_max: float = ALPHA_MAX_DEFAULT,
) -> list[AlphaSphere]:
    """Alpha spheres with radius in ``[alpha_min, alpha_max]``.

    Parameters
    ----------
    coords:
        ``(n_atoms, 3)`` coordinates, Å.
    alpha_min, alpha_max:
        The intermediate band. Spheres smaller than ``alpha_min`` sit in the
        packed interior; larger than ``alpha_max`` they have escaped to bulk
        solvent.
    """
    from scipy.spatial import Delaunay  # noqa: PLC0415

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (n_atoms, 3), got {coords.shape}")
    if len(coords) < 4:
        raise ValueError(
            f"need at least 4 atoms to form a tetrahedron, got {len(coords)}"
        )
    if alpha_min >= alpha_max:
        raise ValueError(f"alpha_min must be < alpha_max, got {alpha_min} >= {alpha_max}")

    # The triangulation is the expensive step and its simplex table is
    # O(n_atoms) tetrahedra, each with 4 indices.
    check_resources(len(coords), 8, dtype=np.float64, label="Delaunay triangulation")

    try:
        tri = Delaunay(coords)
    except Exception as exc:  # QhullError on degenerate input
        raise ValueError(
            f"Delaunay triangulation failed ({exc}). This usually means the atoms "
            f"are coplanar or contain duplicates -- check for a flat or truncated "
            f"structure."
        ) from exc

    out: list[AlphaSphere] = []
    for simplex in tri.simplices:
        res = _circumsphere(coords[simplex])
        if res is None:
            continue
        centre, radius = res
        if alpha_min <= radius <= alpha_max:
            out.append(
                AlphaSphere(
                    centre=centre,
                    radius=radius,
                    atom_indices=tuple(int(i) for i in simplex),
                )
            )
    _log.debug(
        "alpha spheres: %d of %d tetrahedra fell in [%.1f, %.1f] A",
        len(out),
        len(tri.simplices),
        alpha_min,
        alpha_max,
    )
    return out


def _union_volume(
    spheres: list[AlphaSphere],
    *,
    n_samples: int = 20000,
    seed: int = 0,
    box_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Monte Carlo union volume of overlapping spheres.

    Alpha spheres in a pocket overlap heavily, so summing 4/3 pi r^3 would
    overcount severely — often by several-fold. Exact union volume of
    arbitrarily overlapping spheres is an inclusion-exclusion problem that
    blows up combinatorially, so sampling is the pragmatic choice. Relative
    standard error is ~sqrt((1-f)/(f * n_samples)) for occupied fraction f,
    i.e. roughly 1% at the default — around 30 Å³ on a 4500 Å³ cavity.

    ``box_bounds`` forces a specific integration box. Passing the *same*
    bounds and the same ``seed`` for two related volumes makes them share
    sample points, which turns their difference into a **paired** estimate.
    That matters enormously: the independent-sample error on a volume
    difference is dominated by the two volumes' own noise, which can exceed
    the effect being measured. See ``pocket_volume_delta_on_mutation``.
    """
    if not spheres:
        return 0.0
    centres = np.array([s.centre for s in spheres])
    radii = np.array([s.radius for s in spheres])

    if box_bounds is None:
        lo = (centres - radii[:, None]).min(axis=0)
        hi = (centres + radii[:, None]).max(axis=0)
    else:
        lo, hi = box_bounds
    box = float(np.prod(hi - lo))
    if box <= 0:
        return 0.0

    rng = np.random.default_rng(seed)
    pts = rng.uniform(lo, hi, size=(n_samples, 3))
    # Chunk the point-to-centre distance matrix: it is n_samples x n_spheres
    # and a large pocket in a big structure makes that sizeable.
    plan = check_resources(n_samples, len(centres), dtype=np.float64, label="pocket volume MC")
    inside = np.zeros(n_samples, dtype=bool)
    step = max(1, plan.chunk_rows)
    for start in range(0, n_samples, step):
        chunk = pts[start : start + step]
        d2 = ((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        inside[start : start + step] = (d2 <= (radii**2)[None, :]).any(axis=1)
    return box * float(inside.mean())


def detect_pockets(
    coords: np.ndarray,
    *,
    alpha_min: float = ALPHA_MIN_DEFAULT,
    alpha_max: float = ALPHA_MAX_DEFAULT,
    min_spheres: int = 8,
    cluster_distance: float = 4.0,
    compute_volume: bool = True,
    seed: int = 0,
) -> list[Pocket]:
    """Detect pockets: alpha spheres, clustered, ranked by size.

    Clustering is single-linkage at ``cluster_distance`` — two spheres join the
    same pocket if their centres are within that distance. Single linkage is
    the right choice here (a pocket is a connected void, and chaining along a
    channel is a feature, not an artefact), but it is also why a
    ``cluster_distance`` set too large merges a pocket with its neighbours
    through a narrow neck.

    ``min_spheres`` discards specks: a cluster of two or three spheres is
    usually interstitial space, not a site a ligand could occupy.

    .. note::
       This is a compact reimplementation of fpocket's geometric core, not a
       replacement for it. fpocket adds polarity-aware filtering, a
       druggability score and far more validation. Prefer it where available;
       use this where a compiled dependency is unwelcome.
    """
    spheres = alpha_spheres(coords, alpha_min=alpha_min, alpha_max=alpha_max)
    if not spheres:
        _log.info("no alpha spheres in [%.1f, %.1f] A -- no pockets found", alpha_min, alpha_max)
        return []

    centres = np.array([s.centre for s in spheres])
    labels = _single_linkage(centres, cluster_distance)

    pockets: list[Pocket] = []
    for label in np.unique(labels):
        members = [s for s, lab in zip(spheres, labels, strict=True) if lab == label]
        if len(members) < min_spheres:
            continue
        member_centres = np.array([m.centre for m in members])
        pockets.append(
            Pocket(
                centre=member_centres.mean(axis=0),
                spheres=members,
                volume=_union_volume(members, seed=seed) if compute_volume else 0.0,
                atom_indices=frozenset(i for m in members for i in m.atom_indices),
            )
        )

    pockets.sort(key=lambda p: -p.n_spheres)
    for rank, pocket in enumerate(pockets):
        pocket.rank = rank
    _log.info(
        "detected %d pocket(s) from %d alpha spheres; largest has %d spheres, %.1f A^3",
        len(pockets),
        len(spheres),
        pockets[0].n_spheres if pockets else 0,
        pockets[0].volume if pockets else 0.0,
    )
    return pockets


def _single_linkage(points: np.ndarray, cutoff: float) -> np.ndarray:
    """Single-linkage clustering by union-find over the radius-cutoff graph.

    A KD-tree pair query plus union-find is O(n log n + pairs) and avoids
    materialising an n x n distance matrix, which for a large structure's
    alpha spheres would be the single biggest allocation in this module.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i, j in cKDTree(points).query_pairs(cutoff):
        union(i, j)
    return np.array([find(i) for i in range(n)])


# ---------------------------------------------------------------------------
# The three Tier 3 columns
# ---------------------------------------------------------------------------


def distance_to_pocket(ca_coord: np.ndarray, pocket: Pocket) -> float:
    """Cα-to-pocket-centroid distance, Å — the ``dist_to_pocket`` column.

    Centroid distance, not minimum distance to the pocket surface. The centroid
    is the more stable statistic: minimum-distance is dominated by whichever
    single alpha sphere happens to protrude furthest toward the residue, and
    moves substantially between near-identical structures.
    """
    return float(np.linalg.norm(np.asarray(ca_coord, dtype=float) - pocket.centre))


def pocket_volume_delta_on_mutation(
    coords: np.ndarray,
    *,
    sidechain_atom_indices: list[int],
    cb_coord: np.ndarray,
    wt_aa: str,
    mut_aa: str,
    pocket_rank: int = 0,
    seed: int = 0,
    n_samples: int = 60000,
    **detect_kwargs: object,
) -> float:
    """Change in pocket volume on mutation, Å³ — the ``pocket_vol_delta`` column.

    Positive = the pocket **grows** (e.g. Trp→Ala removes bulk).

    .. warning::
       **The SIGN of this quantity is not a reliable steric proxy, and this
       function should not be your source for the** ``pocket_vol_delta``
       **column if you can obtain a repacked one.**

       Measured empirically on constructed geometries during development:
       removing a tryptophan from a cavity produced a *negative* delta
       (pocket measured smaller) across cavity sizes from ~1000 to ~4500 A^3,
       stable under 10x more Monte Carlo samples, i.e. not noise.

       The reason is intrinsic to alpha-sphere detection, not a defect in this
       implementation. Alpha-sphere volume is **band-limited**: only spheres
       with radius in ``[alpha_min, alpha_max]`` count. A side chain
       protruding into a void is itself pocket-lining surface, and it
       subdivides that void into narrower spaces that fall *inside* the band.
       Delete it and the space opens up, spheres grow past ``alpha_max``, and
       the *detected* pocket shrinks even though the physical cavity grew.

       So this measures "change in detected alpha-sphere pocket volume", which
       is well defined, deterministic and reproducible -- but is **not** the
       same quantity as "change in cavity volume", and does not monotonically
       track side-chain size.

       Properly, the volume change on mutation requires repacking the side
       chain and relaxing the structure. Use Rosetta or FoldX and supply
       ``pocket_vol_delta`` in ``variants.csv`` directly. This function exists
       as a dependency-light geometric fallback and is logged at WARNING level
       each time it is called, so the caveat cannot be lost.

       What it *does* capture: the wild-type side-chain atoms are removed and
       replaced by a sphere of the mutant residue's characteristic volume at
       Cbeta, then pockets are redetected. No backbone shift, no rotamer
       selection, no induced fit, no solvent reorganisation.

       The two volumes are integrated as a PAIRED Monte Carlo estimate -- one
       shared box, one shared set of sample points, so the sampling error is
       common to both and largely cancels in the difference. This is what makes
       a single side chain's effect measurable against a cavity orders of
       magnitude larger than it. Raise ``n_samples`` if the pocket is very
       large relative to the substitution; a difference smaller than roughly
       ``pocket_volume / sqrt(n_samples)`` should still not be interpreted.
    """
    for aa in (wt_aa, mut_aa):
        if aa not in SIDECHAIN_VOLUME:
            raise ValueError(f"unknown amino acid {aa!r}; expected one of {sorted(SIDECHAIN_VOLUME)}")

    _log.warning(
        "pocket_volume_delta_on_mutation(%s->%s): alpha-sphere volume is "
        "BAND-LIMITED, so the sign of this delta does not reliably track "
        "side-chain size -- an occluding side chain creates pocket-lining "
        "surface of its own. Prefer a repacked pocket_vol_delta from Rosetta "
        "or FoldX. See the function docstring.",
        wt_aa, mut_aa,
    )

    coords = np.asarray(coords, dtype=float)
    # Volumes are recomputed below as a PAIRED estimate, so skip the
    # per-pocket Monte Carlo here.
    before = detect_pockets(coords, seed=seed, compute_volume=False, **detect_kwargs)  # type: ignore[arg-type]
    if len(before) <= pocket_rank:
        _log.warning(
            "no pocket of rank %d in the wild-type structure; returning 0.0", pocket_rank
        )
        return 0.0

    keep = np.setdiff1d(np.arange(len(coords)), np.asarray(sidechain_atom_indices, dtype=int))
    stripped = coords[keep]

    # Represent the mutant side chain as a sphere of its characteristic volume,
    # placed at Cb. Approximated as a small shell of points so the Delaunay
    # triangulation sees an object with extent rather than a single vertex.
    r_mut = (3.0 * SIDECHAIN_VOLUME[mut_aa] / (4.0 * np.pi)) ** (1.0 / 3.0)
    if mut_aa == "G":
        mutant_atoms = np.zeros((0, 3))  # glycine has no side chain
    else:
        mutant_atoms = _fibonacci_shell(np.asarray(cb_coord, dtype=float), r_mut, n=12)

    mutated = np.vstack([stripped, mutant_atoms]) if len(mutant_atoms) else stripped
    after = detect_pockets(mutated, seed=seed, compute_volume=False, **detect_kwargs)  # type: ignore[arg-type]

    wt_pocket = before[pocket_rank]
    if not after:
        return float(-_union_volume(wt_pocket.spheres, n_samples=n_samples, seed=seed))

    # Match by proximity to the original centroid rather than by rank: a
    # mutation that changes pocket sizes can reorder the ranking, and
    # comparing rank 0 to rank 0 across that reordering compares two
    # different pockets.
    target = wt_pocket.centre
    mut_pocket = min(after, key=lambda p: float(np.linalg.norm(p.centre - target)))

    # PAIRED Monte Carlo. Both volumes are integrated over one shared box with
    # one shared set of sample points, so the sampling error is common to both
    # and largely cancels in the difference. Estimating them independently
    # gives each a ~1% error that can swamp the effect of a single side chain:
    # a tryptophan is ~228 A^3 against a cavity that may be thousands.
    all_spheres = wt_pocket.spheres + mut_pocket.spheres
    lo = np.min([s.centre - s.radius for s in all_spheres], axis=0)
    hi = np.max([s.centre + s.radius for s in all_spheres], axis=0)
    bounds = (lo, hi)

    v_before = _union_volume(wt_pocket.spheres, n_samples=n_samples, seed=seed, box_bounds=bounds)
    v_after = _union_volume(mut_pocket.spheres, n_samples=n_samples, seed=seed, box_bounds=bounds)
    return float(v_after - v_before)


def _fibonacci_shell(centre: np.ndarray, radius: float, n: int = 12) -> np.ndarray:
    """``n`` approximately-equidistant points on a sphere (Fibonacci lattice).

    Used to give a substituted side chain spatial extent. The Fibonacci
    construction gives near-uniform coverage for any ``n``, unlike a
    latitude/longitude grid which bunches at the poles.
    """
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return centre + radius * np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def residue_contacts_ligand(
    residue_atom_coords: np.ndarray,
    ligand_atom_coords: np.ndarray,
    *,
    cutoff: float = 4.5,
    sec_struct: str | None = None,
    require_loop: bool = True,
) -> bool:
    """``contacts_ligand`` — residue in a loop contacting the ligand.

    A contact is any heavy-atom pair within ``cutoff`` (4.5 Å is the usual
    convention: it spans a van der Waals contact plus a little slack, and is
    generous enough to catch a hydrogen bond at ~2.8–3.2 Å).

    ``require_loop`` encodes the specific thing the Tier 3 feature is after.
    A loop residue that touches the ligand is the classic *plastic* site:
    tolerant of substitution because it is not holding a secondary-structure
    element together, yet able to reshape the pocket because it lines it. A
    helix residue making the same contact is a different proposition. Pass
    ``require_loop=False`` for a plain contact test.
    """
    res = np.asarray(residue_atom_coords, dtype=float)
    lig = np.asarray(ligand_atom_coords, dtype=float)
    if res.ndim != 2 or res.shape[1] != 3:
        raise ValueError(f"residue_atom_coords must be (n, 3), got {res.shape}")
    if lig.ndim != 2 or lig.shape[1] != 3:
        raise ValueError(f"ligand_atom_coords must be (n, 3), got {lig.shape}")
    if len(res) == 0 or len(lig) == 0:
        return False

    if require_loop and sec_struct is not None and sec_struct.upper() != "L":
        return False

    check_resources(len(res), len(lig), dtype=np.float64, label="residue-ligand contact matrix")
    d2 = ((res[:, None, :] - lig[None, :, :]) ** 2).sum(axis=2)
    return bool((d2 <= cutoff**2).any())
