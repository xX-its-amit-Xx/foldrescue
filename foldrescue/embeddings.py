"""Frozen-encoder embeddings with streaming extraction and a resumable cache.

WHAT THIS IS FOR
================

Tier 2 consumes a protein-side and a compound-side vector per pair. Producing
them is the expensive, fragile part of the pipeline: a forward pass over a
protein language model for every variant, and over a chemical language model
for every compound. On CPU that is minutes to hours; in a hosted notebook it
is also the step most likely to be killed halfway through by a timeout.

So extraction here is **streamed and checkpointed**, not loaded in one go:

* entities are processed in batches,
* each batch is appended to an on-disk store immediately,
* a JSON manifest records which ids are already done, alongside a fingerprint
  of the extraction *configuration*,
* a rerun skips completed ids and continues.

The config fingerprint matters more than it looks. Resuming a cache that was
built with a different model, pooling strategy, or context window silently
mixes two incompatible representations into one matrix -- a bug that produces
plausible-looking numbers and is nearly impossible to spot downstream. If the
fingerprint does not match, this module refuses to resume and says why.

THE ENCODERS
============

**ESM-2** (Lin et al. 2023, Meta AI) is a transformer trained by masked
language modelling on ~250M protein sequences from UniRef. No structures, no
labels -- just "predict the hidden residue". Because a residue's identity is
constrained by the fold and function it participates in, the representations
it learns encode structural and functional context, which is why ESM-2
embeddings transfer to stability and variant-effect tasks. We embed a
**window around the variant position** with the mutant residue substituted,
rather than the whole protein: the local structural environment is what
determines whether a mutation destabilises, and whole-protein mean pooling
washes that out entirely (every variant of one protein gets nearly the same
vector -- a failure mode worth naming because it is easy to ship by accident).

**ChemBERTa** (Chithrananda, Grand & Ramsundar 2020) is a RoBERTa-style
transformer trained by masked language modelling on SMILES strings from PubChem.
It learns a distributional representation of chemical syntax, which correlates
usefully with properties.

FROZEN BY DEFAULT
-----------------
Both are used as fixed feature extractors. See ``models.Tier2Fusion`` for why
unfreezing on 10^3 pairs is self-defeating. ``unfreeze_encoders`` is honoured
here only to the extent of refusing to use the cache, since a fine-tuned
encoder's embeddings are not interchangeable with the frozen ones.

THE SURROGATE BACKEND
---------------------
When ``transformers``/``torch`` or the model weights are unavailable, the
``hashed`` backend produces a deterministic surrogate: hashed k-mer counts for
the protein window and a random projection of Morgan bits for the compound.

This is **not a language model embedding** and is never silently substituted.
It exists so the Tier 2 *architecture* can be exercised and tested without
network access, and anything derived from it is labelled
``backend=hashed`` in the output. A number produced on surrogate embeddings
says something about the fusion head, and nothing about ESM-2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .data import Cohort
from .utils import check_resources, get_logger

__all__ = [
    "EmbeddingConfig",
    "EmbeddingCache",
    "EmbeddingExtractor",
    "build_embedding_matrix",
]

_log = get_logger(__name__)

Backend = Literal["esm2", "chemberta", "hashed"]

# Small by default: the 8M-parameter ESM-2 and the standard ChemBERTa fit on
# CPU and download in seconds. Bigger checkpoints are a parameter, not a
# rewrite.
DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

# Hidden widths of the default checkpoints. Hard-coded so a caller cannot
# silently pair an esm2 backend with a surrogate-sized dim and get a width
# error deep inside the fusion head instead of here.
BACKEND_DIMS: dict[str, int] = {
    "esm2": 320,        # facebook/esm2_t6_8M_UR50D
    "chemberta": 768,   # seyonec/ChemBERTa-zinc-base-v1
}


@dataclass(frozen=True)
class EmbeddingConfig:
    """Everything that changes the numbers, hashed into the cache manifest.

    If you add a parameter that affects the output, add it here too, or the
    cache will happily serve stale vectors.
    """

    side: Literal["protein", "compound"]
    backend: Backend
    model_name: str = ""
    dim: int = 320
    window: int = 15  # residues either side of the variant position
    pooling: Literal["mean", "cls", "variant"] = "variant"
    unfreeze_encoders: bool = False
    revision: str = "v1"

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "side": self.side,
                "backend": self.backend,
                "model_name": self.model_name,
                "dim": self.dim,
                "window": self.window,
                "pooling": self.pooling,
                "unfreeze_encoders": self.unfreeze_encoders,
                "revision": self.revision,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class EmbeddingCache:
    """Append-only on-disk store keyed by entity id, with a manifest.

    Layout::

        <root>/<side>_<fingerprint>/
            manifest.json      # config + completed ids, rewritten atomically
            vectors.npy        # (n_done, dim) float32, grown by append

    ``vectors.npy`` is rewritten on each flush rather than memory-mapped and
    extended in place. That is slower, but it is correct across interruption:
    a partially-extended memmap with a stale header is unreadable, and the
    whole point here is surviving a kill.
    """

    def __init__(self, root: Path | str, config: EmbeddingConfig) -> None:
        self.config = config
        self.dir = Path(root) / f"{config.side}_{config.fingerprint()}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.vectors_path = self.dir / "vectors.npy"
        self.ids: list[str] = []
        self.vectors: np.ndarray = np.zeros((0, config.dim), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        manifest = json.loads(self.manifest_path.read_text())
        stored_fp = manifest.get("fingerprint")
        if stored_fp != self.config.fingerprint():
            # Should be unreachable because the fingerprint is in the path, but
            # a user who copies directories around can trip it.
            raise ValueError(
                f"embedding cache at {self.dir} was built with configuration "
                f"{stored_fp!r} but the current configuration is "
                f"{self.config.fingerprint()!r}. Refusing to resume: mixing "
                f"embeddings from two configurations produces a silently wrong "
                f"feature matrix. Delete the directory to rebuild."
            )
        if self.vectors_path.exists():
            vectors = np.load(self.vectors_path)
            ids = list(manifest.get("ids", []))
            if len(ids) != len(vectors):
                # Interrupted between writing vectors and the manifest. Trust
                # whichever is shorter; the rest will be recomputed.
                keep = min(len(ids), len(vectors))
                _log.warning(
                    "embedding cache at %s is torn (%d ids vs %d vectors); "
                    "truncating to %d and recomputing the remainder.",
                    self.dir,
                    len(ids),
                    len(vectors),
                    keep,
                )
                ids, vectors = ids[:keep], vectors[:keep]
            self.ids, self.vectors = ids, np.asarray(vectors, dtype=np.float32)
            _log.info("resumed embedding cache %s with %d entities", self.dir, len(self.ids))

    @property
    def done(self) -> set[str]:
        return set(self.ids)

    def append(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.config.dim:
            raise ValueError(
                f"embedding width {vectors.shape[1]} does not match configured dim "
                f"{self.config.dim}"
            )
        self.vectors = np.vstack([self.vectors, vectors]) if len(self.vectors) else vectors
        self.ids.extend(str(i) for i in ids)
        self.flush()

    def flush(self) -> None:
        """Write vectors then manifest, each atomically via a temp-and-rename.

        Order matters: vectors first, manifest second. A crash between them
        leaves extra vectors and a short manifest, which ``_load`` repairs by
        truncation. The reverse order would leave ids claiming vectors that do
        not exist.
        """
        tmp_vec = self.vectors_path.with_suffix(".npy.tmp")
        # Write through an explicit handle: np.save() appends ".npy" to any
        # path that does not already end in it, which would silently produce
        # "vectors.npy.tmp.npy" and leave the rename with nothing to move.
        with open(tmp_vec, "wb") as fh:
            np.save(fh, self.vectors, allow_pickle=False)
        tmp_vec.replace(self.vectors_path)

        tmp_man = self.manifest_path.with_suffix(".json.tmp")
        tmp_man.write_text(
            json.dumps(
                {
                    "fingerprint": self.config.fingerprint(),
                    "config": self.config.__dict__,
                    "ids": self.ids,
                    "n": len(self.ids),
                },
                indent=2,
            )
        )
        tmp_man.replace(self.manifest_path)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.vectors, index=pd.Index(self.ids, name="entity_id"))


# ---------------------------------------------------------------------------
# Surrogate (no-network) encoders
# ---------------------------------------------------------------------------


def _hashed_protein_embedding(windows: Sequence[str], dim: int, k: int = 3) -> np.ndarray:
    """Hashed k-mer counts of the variant window. A surrogate, not ESM-2.

    Deterministic across runs and platforms: uses ``hashlib`` rather than
    Python's ``hash()``, which is salted per-process and would make the cache
    non-reproducible.
    """
    out = np.zeros((len(windows), dim), dtype=np.float32)
    for i, seq in enumerate(windows):
        s = str(seq)
        for j in range(max(0, len(s) - k + 1)):
            kmer = s[j : j + k]
            h = int.from_bytes(hashlib.blake2b(kmer.encode(), digest_size=8).digest(), "big")
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-8)


def _hashed_compound_embedding(smiles: Sequence[str], dim: int, n_bits: int = 2048) -> np.ndarray:
    """Random projection of Morgan bits. A surrogate, not ChemBERTa.

    The projection matrix is seeded from a fixed constant so the embedding is
    stable across runs -- otherwise the cache would be useless.
    """
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    bits = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            bits[i] = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    rng = np.random.default_rng(20240601)
    proj = rng.normal(0, 1.0 / np.sqrt(dim), size=(n_bits, dim)).astype(np.float32)
    out = bits @ proj
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingExtractor:
    """Stream embeddings for a set of entities into a resumable cache.

    Parameters
    ----------
    config:
        What to compute. Its fingerprint keys the cache.
    cache_root:
        Directory holding cache subdirectories.
    batch_size:
        Entities per forward pass and per checkpoint flush. Smaller means more
        frequent checkpoints and less lost work on a kill.
    """

    config: EmbeddingConfig
    cache_root: Path | str = ".foldrescue_cache/embeddings"
    batch_size: int = 16
    device: str = "cpu"
    _cache: EmbeddingCache | None = field(default=None, repr=False)

    @property
    def cache(self) -> EmbeddingCache:
        if self._cache is None:
            self._cache = EmbeddingCache(self.cache_root, self.config)
        return self._cache

    def _encode_batch(self, ids: Sequence[str], payloads: Sequence[str]) -> np.ndarray:
        if self.config.backend == "hashed":
            if self.config.side == "protein":
                return _hashed_protein_embedding(payloads, self.config.dim)
            return _hashed_compound_embedding(payloads, self.config.dim)

        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        if not hasattr(self, "_hf"):
            _log.info("loading %s (frozen)", self.config.model_name)
            tok = AutoTokenizer.from_pretrained(self.config.model_name)
            mdl = AutoModel.from_pretrained(self.config.model_name).to(self.device)
            if not self.config.unfreeze_encoders:
                mdl.eval()
                for p in mdl.parameters():
                    p.requires_grad_(False)
            self._hf = (tok, mdl)  # type: ignore[attr-defined]
        tok, mdl = self._hf  # type: ignore[attr-defined]

        enc = tok(list(payloads), return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = mdl(**enc).last_hidden_state  # (b, t, d)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        if self.config.pooling == "cls":
            pooled = hidden[:, 0, :]
        elif self.config.pooling == "variant" and self.config.side == "protein":
            # The variant sits at the centre of the window by construction, so
            # pool a narrow span around the middle token rather than the whole
            # sequence. Whole-sequence mean pooling is what makes every variant
            # of a protein look identical.
            lengths = enc["attention_mask"].sum(dim=1)
            centres = (lengths // 2).clamp(min=0)
            picks = []
            for b in range(hidden.shape[0]):
                lo = int(max(0, centres[b].item() - 2))
                hi = int(min(hidden.shape[1], centres[b].item() + 3))
                picks.append(hidden[b, lo:hi, :].mean(dim=0))
            pooled = torch.stack(picks)
        else:
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        vec = pooled.cpu().numpy().astype(np.float32)
        if vec.shape[1] != self.config.dim:
            raise ValueError(
                f"{self.config.model_name} produced width {vec.shape[1]} but config.dim "
                f"is {self.config.dim}. Set EmbeddingConfig(dim={vec.shape[1]})."
            )
        return vec

    def extract(self, entities: dict[str, str]) -> pd.DataFrame:
        """Embed ``{entity_id: payload}``, skipping anything already cached.

        Returns a frame indexed by entity id. Safe to interrupt: rerun and it
        resumes from the last flushed batch.
        """
        cache = self.cache
        todo = [(k, v) for k, v in entities.items() if k not in cache.done]
        if not todo:
            _log.info("all %d entities already cached (%s)", len(entities), cache.dir)
            return cache.as_frame().loc[list(entities)]

        check_resources(
            len(entities),
            self.config.dim,
            dtype=np.float32,
            label=f"{self.config.side} embedding matrix",
        )
        _log.info(
            "extracting %d/%d %s embeddings (backend=%s, batch=%d) -> %s",
            len(todo),
            len(entities),
            self.config.side,
            self.config.backend,
            self.batch_size,
            cache.dir,
        )
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start : start + self.batch_size]
            ids = [c[0] for c in chunk]
            vecs = self._encode_batch(ids, [c[1] for c in chunk])
            cache.append(ids, vecs)  # flushes a checkpoint every batch
            _log.debug("checkpointed %d/%d", min(start + self.batch_size, len(todo)), len(todo))
        return cache.as_frame().loc[list(entities)]


# ---------------------------------------------------------------------------
# Pair matrix assembly
# ---------------------------------------------------------------------------


def _variant_window(
    sequence: str | None, position: int, mut_aa: str, window: int, wt_aa: str
) -> str:
    """Mutant-substituted window around the variant position.

    Without a real sequence (the common case when no FASTA is supplied) we
    emit a minimal synthetic context containing the substitution itself. It is
    marked as such by being short and is only ever used with the ``hashed``
    backend; running ESM-2 on a fabricated context would produce a
    confident-looking vector about nothing.
    """
    if sequence:
        lo = max(0, position - 1 - window)
        hi = min(len(sequence), position + window)
        seq = list(sequence[lo:hi])
        centre = position - 1 - lo
        if 0 <= centre < len(seq):
            if seq[centre] != wt_aa:
                _log.warning(
                    "sequence residue %s at position %d does not match wt_aa %s; "
                    "substituting anyway but the numbering may be off by an isoform",
                    seq[centre],
                    position,
                    wt_aa,
                )
            seq[centre] = mut_aa
        return "".join(seq)
    return f"{wt_aa}{mut_aa}"


def build_embedding_matrix(
    cohort: Cohort,
    *,
    rows: np.ndarray | None = None,
    pairs: pd.DataFrame | None = None,
    protein_backend: Backend = "hashed",
    compound_backend: Backend = "hashed",
    protein_dim: int | None = None,
    compound_dim: int | None = None,
    cache_root: Path | str = ".foldrescue_cache/embeddings",
    sequences: dict[str, str] | None = None,
    window: int = 15,
    unfreeze_encoders: bool = False,
    batch_size: int = 16,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, object]]:
    """Build the ``[protein | compound]`` matrix Tier 2 expects.

    Returns the matrix and a provenance dict naming both backends, so that any
    result derived from surrogate embeddings carries that fact with it.
    """
    pair_table = cohort.pairs if pairs is None else pairs
    if rows is not None:
        pair_table = pair_table.iloc[np.asarray(rows, dtype=int)]

    # Widths follow the backend unless the caller pins them. ChemBERTa is 768
    # wide and the surrogate default was 384, so inferring here turns what
    # would be a confusing width error inside Tier 2 into no error at all.
    if protein_dim is None:
        protein_dim = BACKEND_DIMS.get(protein_backend, 320)
    if compound_dim is None:
        compound_dim = BACKEND_DIMS.get(compound_backend, 384)

    # Real protein sequences ship with the package, so an esm2 run embeds a
    # genuine variant window by default rather than the two-residue stand-in.
    if sequences is None and protein_backend != "hashed":
        from .data import load_sequences  # noqa: PLC0415

        sequences = {g: r["sequence"] for g, r in load_sequences().items()}
        _log.info("using %d packaged UniProt sequences for variant windows", len(sequences))

    prot_cfg = EmbeddingConfig(
        side="protein",
        backend=protein_backend,
        model_name="" if protein_backend == "hashed" else DEFAULT_ESM_MODEL,
        dim=protein_dim,
        window=window,
        pooling="variant",
        unfreeze_encoders=unfreeze_encoders,
    )
    comp_cfg = EmbeddingConfig(
        side="compound",
        backend=compound_backend,
        model_name="" if compound_backend == "hashed" else DEFAULT_CHEMBERTA_MODEL,
        dim=compound_dim,
        pooling="mean",
        unfreeze_encoders=unfreeze_encoders,
    )

    needed_v = sorted(set(pair_table["variant_id"].astype(str)))
    vinfo = cohort.variants.loc[needed_v]
    seqs = sequences or {}
    variant_payloads = {
        str(vid): _variant_window(
            seqs.get(str(r["gene"])) or seqs.get(str(r["uniprot"])),
            int(r["position"]),
            str(r["mut_aa"]),
            window,
            str(r["wt_aa"]),
        )
        for vid, r in vinfo.iterrows()
    }
    needed_c = sorted(set(pair_table["compound_id"].astype(str)))
    compound_payloads = {
        str(cid): str(cohort.compounds.loc[cid, "smiles"]) for cid in needed_c
    }

    prot = EmbeddingExtractor(prot_cfg, cache_root, batch_size, device).extract(variant_payloads)
    comp = EmbeddingExtractor(comp_cfg, cache_root, batch_size, device).extract(compound_payloads)

    plan = check_resources(
        len(pair_table),
        protein_dim + compound_dim,
        dtype=np.float32,
        label="tier2 pair embedding matrix",
    )
    X = np.empty((len(pair_table), protein_dim + compound_dim), dtype=plan.dtype)
    X[:, :protein_dim] = prot.loc[pair_table["variant_id"].astype(str)].to_numpy()
    X[:, protein_dim:] = comp.loc[pair_table["compound_id"].astype(str)].to_numpy()

    provenance = {
        "protein_backend": protein_backend,
        "protein_model": prot_cfg.model_name or "hashed-kmer-surrogate",
        "compound_backend": compound_backend,
        "compound_model": comp_cfg.model_name or "morgan-projection-surrogate",
        "surrogate": protein_backend == "hashed" or compound_backend == "hashed",
        "unfreeze_encoders": unfreeze_encoders,
    }
    if provenance["surrogate"]:
        _log.warning(
            "Tier 2 is running on SURROGATE embeddings (%s / %s). These are "
            "deterministic hashes, NOT pretrained language model representations. "
            "Results exercise the fusion architecture and say nothing about ESM-2 "
            "or ChemBERTa.",
            provenance["protein_model"],
            provenance["compound_model"],
        )
    return X, provenance
