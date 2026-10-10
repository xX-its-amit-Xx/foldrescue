"""The three model tiers, behind one interface.

WHY THREE TIERS
===============

Because the comparison is the result. "We built a cross-attention transformer
fusion model" is not a finding; "a cross-attention fusion model does not beat
gradient-boosted trees on 10^3 pairs" is. Reporting only the fanciest
architecture that happened to work is how a field accumulates methods nobody
can reproduce.

``Tier1GBM`` -- interpretable baseline
    LightGBM over the concatenated feature blocks. Histogram-based gradient
    boosting (Ke et al. 2017): each tree splits on binned features, and the
    leaf-wise growth strategy means it finds deep interactions in few trees.
    On tabular data at n ~ 10^2-10^4 this is the model to beat, and it usually
    is not beaten. It also gives per-feature attribution directly.

``Tier2Fusion`` -- representation fusion
    A frozen protein language model embedding of the variant sequence context
    and a frozen chemical language model embedding of the compound, joined by a
    cross-attention block and an MLP head.

    Why frozen by default: ESM-2 650M has ~650 million parameters; a benchmark
    of ~10^3 labelled pairs supports fitting perhaps 10^4. Unfreezing is
    guaranteed overfitting -- not likely, guaranteed, in the sense that the
    model can memorise the training set exactly. The ``unfreeze_encoders``
    flag exists because reviewers ask for it, and it logs a warning that says
    this.

    Why cross-attention rather than concatenation: concatenating two
    embeddings and passing them to an MLP forces the first layer to learn all
    pair interactions as a linear map of the joint vector. Cross-attention
    (Vaswani et al. 2017, applied across two modalities) lets each compound
    token query the protein representation directly, which is a better
    inductive bias for "which part of this molecule matters for this
    residue?" -- the mechanism the task is actually about.

``Tier3StructureAware`` -- pocket occupancy
    Tier 1 plus the structural block: distance from the variant residue to the
    chaperone binding pocket, pocket volume change on mutation, and whether
    the residue sits in a loop contacting the ligand. Separated from Tier 1 so
    the ablation can price the structural information, which is the most
    expensive input to obtain (it needs a structure and a pocket definition).

HONEST SMALL-n PRACTICE
-----------------------
Every tier exposes ``fit(X, y)``/``predict_proba(X)`` and nothing else, so the
evaluation harness cannot accidentally give one tier more information than
another. Hyperparameters are fixed conservative defaults rather than tuned per
split: tuning on a 200-row test set and reporting the best number is a way of
fitting the test set by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .utils import check_resources, get_logger

__all__ = [
    "PairModel",
    "Tier1GBM",
    "Tier2Fusion",
    "Tier3StructureAware",
    "build_model",
    "TIER_NAMES",
]

_log = get_logger(__name__)

TIER_NAMES = ("tier1_gbm", "tier2_fusion", "tier3_structure")


@runtime_checkable
class PairModel(Protocol):
    """Minimal interface every tier satisfies."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> PairModel: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


@dataclass
class Tier1GBM:
    """LightGBM over the concatenated feature blocks.

    Defaults are deliberately conservative for small n: shallow-ish trees,
    strong feature and row subsampling, and a minimum leaf count that stops
    the model carving out single-example leaves. ``n_estimators`` is modest and
    paired with early stopping when a validation set is supplied.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.6
    reg_lambda: float = 1.0
    random_state: int = 0
    n_jobs: int | None = None
    scale_pos_weight: float | None = None
    name: str = "tier1_gbm"
    model_: object | None = field(default=None, repr=False)
    feature_names_: list[str] | None = field(default=None, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        feature_names: list[str] | None = None,
        **_: object,
    ) -> Tier1GBM:
        import lightgbm as lgb  # noqa: PLC0415

        X = np.asarray(X)
        y = np.asarray(y).ravel()
        plan = check_resources(
            X.shape[0], X.shape[1], dtype=X.dtype, n_jobs=self.n_jobs, label="tier1 fit"
        )
        self.feature_names_ = feature_names

        params: dict[str, object] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": plan.n_jobs,
            "verbose": -1,
        }
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        clf = lgb.LGBMClassifier(**params)

        callbacks = []
        fit_kw: dict[str, object] = {}
        if eval_set is not None:
            fit_kw["eval_set"] = [(np.asarray(eval_set[0]), np.asarray(eval_set[1]).ravel())]
            fit_kw["eval_metric"] = "average_precision"
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        if callbacks:
            fit_kw["callbacks"] = callbacks

        clf.fit(X.astype(plan.dtype, copy=False), y, **fit_kw)
        self.model_ = clf
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Tier1GBM.fit must be called before predict_proba")
        return np.asarray(self.model_.predict_proba(np.asarray(X))[:, 1], dtype=float)

    def feature_importance(self, kind: str = "gain") -> np.ndarray:
        """Built-in split-gain importance.

        Reported in the cookbook *alongside* permutation and SHAP, never
        instead of them: gain importance is biased toward high-cardinality
        features, which here means the 1024 fingerprint columns are
        structurally favoured over the ~17 stability columns.
        """
        if self.model_ is None:
            raise RuntimeError("fit first")
        booster = self.model_.booster_
        return np.asarray(booster.feature_importance(importance_type=kind), dtype=float)


# ---------------------------------------------------------------------------
# Tier 3 (declared before Tier 2 only because it is a thin Tier 1 variant)
# ---------------------------------------------------------------------------


@dataclass
class Tier3StructureAware(Tier1GBM):
    """Tier 1 with the structural block present.

    Architecturally identical to Tier 1 on purpose. The tier is defined by its
    *inputs*, not its machinery, so that the difference between Tier 1 and
    Tier 3 numbers is attributable to the pocket-occupancy features and
    nothing else. Comparing a GBM to a GBM-with-extra-columns is a clean
    experiment; comparing a GBM to a different architecture with extra columns
    is not.
    """

    name: str = "tier3_structure"
    # Slightly more capacity: the structural block adds genuinely informative
    # low-cardinality columns worth a few more splits.
    num_leaves: int = 23


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------


@dataclass
class Tier2Fusion:
    """Frozen-encoder embeddings + cross-attention + MLP head.

    The model consumes *embeddings*, not raw features: ``fit`` expects an
    ``X`` whose columns are ``[protein_embedding | compound_embedding]``, split
    at ``protein_dim``. ``foldrescue.embeddings`` produces exactly that, with
    streaming extraction and a resumable on-disk cache.

    Parameters
    ----------
    protein_dim, compound_dim:
        Widths of the two embedding blocks in ``X``.
    d_model:
        Width both modalities are projected to before attention.
    n_heads:
        Attention heads. Multi-head attention splits ``d_model`` into
        ``n_heads`` subspaces so different heads can attend to different kinds
        of correspondence simultaneously.
    unfreeze_encoders:
        Has no effect on this class (which never holds the encoders) but is
        recorded and warned about, and is honoured by
        ``foldrescue.embeddings.EmbeddingExtractor``. Kept here so the flag is
        discoverable where users look for it.
    """

    protein_dim: int = 320
    compound_dim: int = 384
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.3
    epochs: int = 120
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 20
    unfreeze_encoders: bool = False
    random_state: int = 0
    device: str = "cpu"
    name: str = "tier2_fusion"
    module_: object | None = field(default=None, repr=False)
    history_: list[dict[str, float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.unfreeze_encoders:
            _log.warning(
                "=" * 78
                + "\nunfreeze_encoders=True. READ THIS.\n"
                "ESM-2 and ChemBERTa have 10^7-10^8 parameters. A pharmacological\n"
                "chaperone benchmark has ~10^3 labelled pairs. Fine-tuning here does not\n"
                "risk overfitting, it guarantees it: the encoders have more than enough\n"
                "capacity to memorise every training pair exactly, and your held-out\n"
                "numbers will collapse while training loss goes to zero.\n"
                "If you are doing this, hold out a gene you never look at until the end.\n"
                + "=" * 78
            )

    def _build(self) -> object:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        prot_dim, comp_dim = self.protein_dim, self.compound_dim
        d_model, n_heads, dropout = self.d_model, self.n_heads, self.dropout

        class CrossAttentionFusion(nn.Module):
            """Compound queries the protein representation, then an MLP head.

            Both embeddings arrive as single pooled vectors, so each modality
            is a sequence of length 1. Attention over length-1 sequences is
            degenerate on its own (softmax over one element is 1), so the
            protein side is first expanded into ``n_slots`` learned views --
            the projection produces several distinct "aspects" of the protein
            representation for the compound to attend over. Without this the
            attention block would reduce to a gated linear layer.
            """

            def __init__(self, n_slots: int = 8) -> None:
                super().__init__()
                self.n_slots = n_slots
                self.prot_proj = nn.Linear(prot_dim, d_model * n_slots)
                self.comp_proj = nn.Linear(comp_dim, d_model)
                self.attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                self.norm_q = nn.LayerNorm(d_model)
                self.norm_kv = nn.LayerNorm(d_model)
                self.norm_out = nn.LayerNorm(d_model)
                self.head = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )

            def forward(self, prot: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
                b = prot.shape[0]
                kv = self.norm_kv(self.prot_proj(prot).view(b, self.n_slots, d_model))
                q = self.norm_q(self.comp_proj(comp)).unsqueeze(1)
                attended, _ = self.attn(q, kv, kv, need_weights=False)
                attended = self.norm_out(attended.squeeze(1))
                # Residual-style concatenation: keep the compound's own
                # representation alongside what it retrieved from the protein,
                # so the head can use either.
                return self.head(torch.cat([attended, q.squeeze(1)], dim=-1)).squeeze(-1)

        torch.manual_seed(self.random_state)
        return CrossAttentionFusion()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        **_: object,
    ) -> Tier2Fusion:
        try:
            import torch  # noqa: PLC0415
            from torch import nn  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Tier 2 requires PyTorch (pip install torch). Tiers 1 and 3 do not; "
                "if you only need the interpretable baseline, use --tier tier1_gbm."
            ) from exc

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        expected = self.protein_dim + self.compound_dim
        if X.shape[1] != expected:
            raise ValueError(
                f"Tier2Fusion expects X with {expected} columns "
                f"(protein_dim={self.protein_dim} + compound_dim={self.compound_dim}), "
                f"got {X.shape[1]}. Build it with foldrescue.embeddings.build_embedding_matrix."
            )
        check_resources(X.shape[0], X.shape[1], dtype=X.dtype, label="tier2 fit")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = torch.device(self.device)
        module = self._build().to(device)

        Xp = torch.from_numpy(X[:, : self.protein_dim]).to(device)
        Xc = torch.from_numpy(X[:, self.protein_dim :]).to(device)
        yt = torch.from_numpy(y).to(device)

        # Class imbalance: weight the positive class by the inverse base rate
        # so the model does not simply learn the majority class.
        pos = float(y.sum())
        pos_weight = torch.tensor(
            [(len(y) - pos) / pos if 0 < pos < len(y) else 1.0], device=device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(module.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        have_eval = eval_set is not None
        if have_eval:
            Xe = np.asarray(eval_set[0], dtype=np.float32)
            ye = np.asarray(eval_set[1], dtype=np.float32).ravel()
            Xep = torch.from_numpy(Xe[:, : self.protein_dim]).to(device)
            Xec = torch.from_numpy(Xe[:, self.protein_dim :]).to(device)
            yet = torch.from_numpy(ye).to(device)

        best_loss, best_state, stale = float("inf"), None, 0
        n = len(y)
        self.history_ = []
        for epoch in range(self.epochs):
            module.train()
            perm = torch.randperm(n, device=device)
            running = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                logits = module(Xp[idx], Xc[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                # Clipping: the attention block can produce large gradients
                # early, and a single bad step wastes a short training run.
                nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                opt.step()
                running += float(loss) * len(idx)
            train_loss = running / n

            if have_eval:
                module.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(module(Xep, Xec), yet))
                self.history_.append(
                    {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                )
                if val_loss < best_loss - 1e-5:
                    best_loss, stale = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                else:
                    stale += 1
                    if stale >= self.patience:
                        _log.info("tier2: early stop at epoch %d (best val %.4f)", epoch, best_loss)
                        break
            else:
                self.history_.append({"epoch": epoch, "train_loss": train_loss})

        if best_state is not None:
            module.load_state_dict(best_state)
        self.module_ = module
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415

        if self.module_ is None:
            raise RuntimeError("Tier2Fusion.fit must be called before predict_proba")
        X = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device)
        self.module_.eval()  # type: ignore[union-attr]
        outs = []
        with torch.no_grad():
            # Chunked inference so a large query set cannot blow up memory.
            for start in range(0, len(X), 4096):
                chunk = X[start : start + 4096]
                p = torch.from_numpy(chunk[:, : self.protein_dim]).to(device)
                c = torch.from_numpy(chunk[:, self.protein_dim :]).to(device)
                outs.append(torch.sigmoid(self.module_(p, c)).cpu().numpy())  # type: ignore[operator]
        return np.concatenate(outs).astype(float)


# ---------------------------------------------------------------------------


def build_model(tier: str, **kwargs: object) -> PairModel:
    """Factory keyed on the tier names used throughout the CLI and cookbook."""
    if tier == "tier1_gbm":
        return Tier1GBM(**kwargs)  # type: ignore[arg-type]
    if tier == "tier3_structure":
        return Tier3StructureAware(**kwargs)  # type: ignore[arg-type]
    if tier == "tier2_fusion":
        return Tier2Fusion(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown tier {tier!r}; valid: {TIER_NAMES}")
